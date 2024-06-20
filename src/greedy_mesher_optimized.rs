use std::{collections::VecDeque, ops::DerefMut as _};

use bevy::{math::ivec3, prelude::*, utils::HashMap};
use bumpalo::{collections::Vec, Bump};
use once_cell::sync::Lazy;

use crate::{
    chunk_mesh::ChunkMesh,
    chunks_refs::ChunksRefs,
    constants::{ADJACENT_AO_DIRS, CHUNK_SIZE, CHUNK_SIZE_P},
    face_direction::FaceDir,
    lod::Lod,
    pool::{Reset, SimplePool},
    utils::{generate_indices, make_vertex_u32, vec3_to_index},
};

struct PooledMemory {
    axis_cols: Box<[[[u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3]>,
    col_face_masks: Box<[[[u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6]>,
}

impl Default for PooledMemory {
    fn default() -> Self {
        // This is much slower than allocating on the stack because the allocation first happens on the stack, then
        // it is copied to the heap.
        Self {
            axis_cols: Box::new([[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3]),
            col_face_masks: Box::new([[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6]),
        }
    }
}

impl Reset for PooledMemory {
    fn reset(mut self) -> Option<Self> {
        for v in self.axis_cols.iter_mut() {
            for v in v.iter_mut() {
                v.fill(0)
            }
        }
        for v in self.col_face_masks.iter_mut() {
            for v in v.iter_mut() {
                v.fill(0)
            }
        }
        Some(self)
    }
}

static MEM_POOL: Lazy<SimplePool<PooledMemory>> = Lazy::new(SimplePool::default);
static HASH_POOL: Lazy<SimplePool<HashMap<u32, [[u32; 32]; CHUNK_SIZE]>>> =
    Lazy::new(SimplePool::default);

pub fn build_chunk_mesh(arena: &Bump, chunks_refs: &ChunksRefs, lod: Lod) -> Option<ChunkMesh> {
    // early exit, if all faces are culled
    if chunks_refs.is_all_voxels_same() {
        return None;
    }
    let mut mesh = ChunkMesh::default();

    let mut mem = MEM_POOL.take();

    // solid binary for each x,y,z axis (3)
    let PooledMemory {
        ref mut axis_cols,
        ref mut col_face_masks,
    } = mem.deref_mut();

    #[inline]
    fn add_voxel_to_axis_cols(
        b: &crate::voxel::BlockData,
        x: usize,
        y: usize,
        z: usize,
        axis_cols: &mut [[[u64; 34]; 34]; 3],
    ) {
        if b.block_type.is_solid() {
            // x,z - y axis
            axis_cols[0][z][x] |= 1u64 << y as u64;
            // z,y - x axis
            axis_cols[1][y][z] |= 1u64 << x as u64;
            // x,y - z axis
            axis_cols[2][y][x] |= 1u64 << z as u64;
        }
    }

    // inner chunk voxels.
    let chunk = &*chunks_refs.chunks[vec3_to_index(IVec3::new(1, 1, 1), 3)];
    assert!(chunk.voxels.len() == CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE || chunk.voxels.len() == 1);
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let i = match chunk.voxels.len() {
                    1 => 0,
                    _ => (z * CHUNK_SIZE + y) * CHUNK_SIZE + x,
                };
                add_voxel_to_axis_cols(&chunk.voxels[i], x + 1, y + 1, z + 1, axis_cols)
            }
        }
    }

    // neighbor chunk voxels.
    // note(leddoo): couldn't be bothered to optimize these.
    //  might be worth it though. together, they take
    //  almost as long as the entire "inner chunk" loop.
    for z in [0, CHUNK_SIZE_P - 1] {
        for y in 0..CHUNK_SIZE_P {
            for x in 0..CHUNK_SIZE_P {
                let pos = ivec3(x as i32, y as i32, z as i32) - IVec3::ONE;
                add_voxel_to_axis_cols(chunks_refs.get_block(pos), x, y, z, axis_cols);
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for y in [0, CHUNK_SIZE_P - 1] {
            for x in 0..CHUNK_SIZE_P {
                let pos = ivec3(x as i32, y as i32, z as i32) - IVec3::ONE;
                add_voxel_to_axis_cols(chunks_refs.get_block(pos), x, y, z, axis_cols);
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for x in [0, CHUNK_SIZE_P - 1] {
            for y in 0..CHUNK_SIZE_P {
                let pos = ivec3(x as i32, y as i32, z as i32) - IVec3::ONE;
                add_voxel_to_axis_cols(chunks_refs.get_block(pos), x, y, z, axis_cols);
            }
        }
    }

    // face culling
    for axis in 0..3 {
        for z in 0..CHUNK_SIZE_P {
            for x in 0..CHUNK_SIZE_P {
                // set if current is solid, and next is air
                let col = axis_cols[axis][z][x];

                // sample descending axis, and set true when air meets solid
                col_face_masks[2 * axis + 0][z][x] = col & !(col << 1);
                // sample ascending axis, and set true when air meets solid
                col_face_masks[2 * axis + 1][z][x] = col & !(col >> 1);
            }
        }
    }

    // greedy meshing planes for every axis (6)
    // key(block + ao) -> HashMap<axis(0-32), binary_plane>
    // note(leddoo): don't ask me how this isn't a massive blottleneck.
    //  might become an issue in the future, when there are more block types.
    //  consider using a single hashmap with key (axis, block_hash, y).
    let mut data: [_; 6] = std::array::from_fn(|_| HASH_POOL.take());

    // find faces and build binary planes based on the voxel block+ao etc...
    for axis in 0..6 {
        let axis_data = &mut data[axis];
        let col_face_mask = &mut col_face_masks[axis];
        for z in 0..CHUNK_SIZE {
            let col_face_mask_z = &mut col_face_mask[z + 1];
            for x in 0..CHUNK_SIZE {
                // skip padded by adding 1(for x padding) and (z+1) for (z padding)
                let mut col = col_face_mask_z[x + 1];

                // removes the right most padding value, because it's invalid
                col >>= 1;
                // removes the left most padding value, because it's invalid
                col &= !(1 << CHUNK_SIZE as u64);

                while col != 0 {
                    let y = col.trailing_zeros();
                    // clear least significant set bit
                    col &= col - 1;

                    // get the voxel position based on axis
                    let voxel_pos = match axis {
                        0 | 1 => ivec3(x as i32, y as i32, z as i32), // down,up
                        2 | 3 => ivec3(y as i32, z as i32, x as i32), // left, right
                        _ => ivec3(x as i32, z as i32, y as i32),     // forward, back
                    };

                    // calculate ambient occlusion
                    let mut ao_index = 0;
                    for (ao_i, ao_offset) in ADJACENT_AO_DIRS.iter().enumerate() {
                        // ambient occlusion is sampled based on axis(ascent or descent)
                        let ao_sample_offset = match axis {
                            0 => ivec3(ao_offset.x, -1, ao_offset.y), // down
                            1 => ivec3(ao_offset.x, 1, ao_offset.y),  // up
                            2 => ivec3(-1, ao_offset.y, ao_offset.x), // left
                            3 => ivec3(1, ao_offset.y, ao_offset.x),  // right
                            4 => ivec3(ao_offset.x, ao_offset.y, -1), // forward
                            _ => ivec3(ao_offset.x, ao_offset.y, 1),  // back
                        };
                        let ao_voxel_pos = voxel_pos + ao_sample_offset;
                        let ao_block = chunks_refs.get_block(ao_voxel_pos);
                        if ao_block.block_type.is_solid() {
                            ao_index |= 1u32 << ao_i;
                        }
                    }

                    let current_voxel = chunks_refs.get_block_no_neighbour(voxel_pos);
                    // let current_voxel = chunks_refs.get_block(voxel_pos);
                    // we can only greedy mesh same block types + same ambient occlusion
                    let block_hash = ao_index | ((current_voxel.block_type as u32) << 9);
                    let data = &mut axis_data.entry(block_hash).or_default()[y as usize];
                    data[x] |= 1u32 << z as u32;
                }
            }
        }
    }

    let mut vertices = vec![];
    for (axis, mut block_ao_data) in data.into_iter().enumerate() {
        let facedir = match axis {
            0 => FaceDir::Down,
            1 => FaceDir::Up,
            2 => FaceDir::Left,
            3 => FaceDir::Right,
            4 => FaceDir::Forward,
            _ => FaceDir::Back,
        };
        for (block_ao, axis_plane) in block_ao_data.iter_mut() {
            let ao = block_ao & 0b111111111;
            let block_type = block_ao >> 9;
            for (axis_pos, plane) in axis_plane.iter_mut().enumerate() {
                let quads_from_axis = greedy_mesh_binary_plane(arena, plane, lod.size() as u32);

                quads_from_axis.into_iter().for_each(|q| {
                    q.append_vertices(
                        &mut vertices,
                        facedir,
                        axis_pos as u32,
                        &Lod::L32,
                        ao,
                        block_type,
                    )
                });
            }
        }
    }

    mesh.vertices.extend(vertices);
    if mesh.vertices.is_empty() {
        None
    } else {
        generate_indices(&mut mesh.indices, mesh.vertices.len());
        Some(mesh)
    }
}

// todo: compress further?
#[derive(Debug)]
pub struct GreedyQuad {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl GreedyQuad {
    ///! compress this quad data into the input vertices vec
    pub fn append_vertices(
        &self,
        vertices: &mut impl Extend<u32>,
        face_dir: FaceDir,
        axis: u32,
        lod: &Lod,
        ao: u32,
        block_type: u32,
    ) {
        // let negate_axis = face_dir.negate_axis();
        // let axis = axis as i32 + negate_axis;
        let axis = axis as i32;
        let jump = lod.jump_index();

        // pack ambient occlusion strength into vertex
        let v1ao = ((ao >> 0) & 1) + ((ao >> 1) & 1) + ((ao >> 3) & 1);
        let v2ao = ((ao >> 3) & 1) + ((ao >> 6) & 1) + ((ao >> 7) & 1);
        let v3ao = ((ao >> 5) & 1) + ((ao >> 8) & 1) + ((ao >> 7) & 1);
        let v4ao = ((ao >> 1) & 1) + ((ao >> 2) & 1) + ((ao >> 5) & 1);

        let v1 = make_vertex_u32(
            face_dir.world_to_sample(axis as i32, self.x as i32, self.y as i32, &lod) * jump,
            v1ao,
            face_dir.normal_index(),
            block_type,
        );
        let v2 = make_vertex_u32(
            face_dir.world_to_sample(
                axis as i32,
                self.x as i32 + self.w as i32,
                self.y as i32,
                &lod,
            ) * jump,
            v2ao,
            face_dir.normal_index(),
            block_type,
        );
        let v3 = make_vertex_u32(
            face_dir.world_to_sample(
                axis as i32,
                self.x as i32 + self.w as i32,
                self.y as i32 + self.h as i32,
                &lod,
            ) * jump,
            v3ao,
            face_dir.normal_index(),
            block_type,
        );
        let v4 = make_vertex_u32(
            face_dir.world_to_sample(
                axis as i32,
                self.x as i32,
                self.y as i32 + self.h as i32,
                &lod,
            ) * jump,
            v4ao,
            face_dir.normal_index(),
            block_type,
        );

        // the quad vertices to be added
        let mut new_vertices = VecDeque::from([v1, v2, v3, v4]);

        // triangle vertex order is different depending on the facing direction
        // due to indices always being the same
        if face_dir.reverse_order() {
            // keep first index, but reverse the rest
            let o = new_vertices.split_off(1);
            o.into_iter().rev().for_each(|i| new_vertices.push_back(i));
        }

        // anisotropy flip
        if (v1ao > 0) ^ (v3ao > 0) {
            // right shift array, to swap triangle intersection angle
            let f = new_vertices.pop_front().unwrap();
            new_vertices.push_back(f);
        }

        vertices.extend(new_vertices);
    }
}

///! generate quads of a binary slice
///! lod not implemented atm
pub fn greedy_mesh_binary_plane<'a, 'b>(
    arena: &'a Bump,
    data: &'b mut [u32; 32],
    lod_size: u32,
) -> Vec<'a, GreedyQuad> {
    let mut greedy_quads = Vec::new_in(arena);
    for row in 0..data.len() {
        let mut y = 0;
        while y < lod_size {
            // find first solid, "air/zero's" could be first so skip
            y += (data[row] >> y).trailing_zeros();
            if y >= lod_size {
                // reached top
                continue;
            }
            let h = (data[row] >> y).trailing_ones();
            // convert height 'num' to positive bits repeated 'num' times aka:
            // 1 = 0b1, 2 = 0b11, 4 = 0b1111
            let h_as_mask = u32::checked_shl(1, h).map_or(!0, |v| v - 1);
            let mask = h_as_mask << y;
            // grow horizontally
            let mut w = 1;
            while row + w < lod_size as usize {
                // fetch bits spanning height, in the next row
                let next_row_h = (data[row + w] >> y) & h_as_mask;
                if next_row_h != h_as_mask {
                    break; // can no longer expand horizontally
                }

                // nuke the bits we expanded into
                data[row + w] = data[row + w] & !mask;

                w += 1;
            }
            greedy_quads.push(GreedyQuad {
                y,
                w: w as u32,
                h,
                x: row as u32,
            });
            y += h;
        }
    }
    greedy_quads
}
