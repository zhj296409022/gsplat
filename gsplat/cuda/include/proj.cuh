#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "types.cuh"
#include "symeigen.cuh"

namespace gsplat {

template <typename T>
inline __device__ void ortho_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x + cx, fy * y + cy});
}

template <typename T>
inline __device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    v_mean3d += vec3<T>(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    // T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    // T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    // T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    // T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    // T rz = 1.f / z;
    // T rz2 = rz * rz;
    // T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    // T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x, max(-lim_x, x * rz));
    T ty = z * min(lim_y, max(-lim_y, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});
}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    // T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    // T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    // T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    // T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    // T rz = 1.f / z;
    // T rz2 = rz * rz;
    // T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    // T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x, max(-lim_x, x * rz));
    T ty = z * min(lim_y, max(-lim_y, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    // if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
    //     v_mean3d.x += -fx * rz2 * v_J[2][0];
    // } else {
    //     v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    // }
    // if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
    //     v_mean3d.y += -fy * rz2 * v_J[2][1];
    // } else {
    //     v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    // }
    if (x * rz <= lim_x && x * rz >= -lim_x) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y && y * rz >= -lim_y) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];
}

template <typename T>
inline __device__ void fisheye_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T eps = 0.0000001f;
    T xy_len = glm::length(glm::vec2({x, y})) + eps;
    T theta = glm::atan(xy_len, z + eps);
    mean2d =
        vec2<T>({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T x2y2z2_inv = 1.f / (x2y2 + z * z);

    T b = glm::atan(xy_len, z) / xy_len / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}

template <typename T>
inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const T eps = 0.0000001f;
    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T len_xy = length(glm::vec2({x, y})) + eps;
    const T x2y2z2 = x2y2 + z * z;
    T x2y2z2_inv = 1.f / x2y2z2;
    T b = glm::atan(len_xy, z) / len_xy / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3<T>(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const T theta = glm::atan(len_xy, z);
    const T J_b = theta / len_xy / x2y2;
    const T J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;
    T l4 = x2y2z2 * x2y2z2;

    T E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    T F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    T A = x * (3 * E + x2 * F);
    T B = y * (E + x2 * F);
    T C = x * (E + y2 * F);
    T D = y * (3 * E + y2 * F);

    T S1 = x2 - y2 - z * z;
    T S2 = y2 - x2 - z * z;
    T inv1 = x2y2z2_inv * x2y2z2_inv;
    T inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    T dJ_dx00 = fx * A * inv2;
    T dJ_dx01 = fx * B * inv2;
    T dJ_dx02 = fx * S1 * inv1;
    T dJ_dx10 = fy * B * inv2;
    T dJ_dx11 = fy * C * inv2;
    T dJ_dx12 = 2.f * fy * xy * inv1;

    T dJ_dy00 = dJ_dx01;
    T dJ_dy01 = fx * C * inv2;
    T dJ_dy02 = 2.f * fx * xy * inv1;
    T dJ_dy10 = dJ_dx11;
    T dJ_dy11 = fy * D * inv2;
    T dJ_dy12 = fy * S2 * inv1;

    T dJ_dz00 = dJ_dx02;
    T dJ_dz01 = dJ_dy02;
    T dJ_dz02 = 2.f * fx * x * z * inv1;
    T dJ_dz10 = dJ_dx12;
    T dJ_dz11 = dJ_dy12;
    T dJ_dz12 = 2.f * fy * y * z * inv1;

    T dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                   dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                   dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    T dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                   dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                   dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    T dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                   dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                   dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}

// rade

template <typename T, bool INTE = false>
inline __device__ bool rade_persp_proj(
    // inputs
    const mat3<T> W, const vec3<T> mean3d, const mat3<T> cov3d, 
    const T fx, const T fy, const T cx,
    const T cy, const uint32_t width, const uint32_t height,
    // outputs
    mat2<T> &cov2d, vec2<T> &mean2d, vec2<T> &ray_plane, vec3<T> &normal,
    T *__restrict__ invraycov3Ds) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T u = min(lim_x, max(-lim_x, x * rz)); // txtz
    T v = min(lim_y, max(-lim_y, y * rz)); // tytz
    T tx = z * u;
    T ty = z * v;

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(fx * rz, 0.f,                  // 1st column
                            0.f, fy * rz,                  // 2nd column
                            -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});

    // calculate the ray space intersection plane.
    auto length = [](T x, T y, T z) { return sqrt(x*x+y*y+z*z); };
    mat3<T> cov3d_eigen_vector;
	vec3<T> cov3d_eigen_value;
	int D = glm_modification::findEigenvaluesSymReal(cov3d,cov3d_eigen_value,cov3d_eigen_vector);
    unsigned int min_id = cov3d_eigen_value[0]>cov3d_eigen_value[1]? (cov3d_eigen_value[1]>cov3d_eigen_value[2]?2:1):(cov3d_eigen_value[0]>cov3d_eigen_value[2]?2:0);
    mat3<T> cov3d_inv;
	bool well_conditioned = cov3d_eigen_value[min_id]>1E-8;
	vec3<T> eigenvector_min;
	if(well_conditioned)
	{
		mat3<T> diag = mat3<T>( 1/cov3d_eigen_value[0], 0, 0,
                                0, 1/cov3d_eigen_value[1], 0,
                                0, 0, 1/cov3d_eigen_value[2] );
		cov3d_inv = cov3d_eigen_vector * diag * glm::transpose(cov3d_eigen_vector);
	}
	else
	{
		eigenvector_min = cov3d_eigen_vector[min_id];
		cov3d_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
    vec3<T> uvh = {u, v, 1};
    vec3<T> Cinv_uvh = cov3d_inv * uvh;
    if(length(Cinv_uvh.x, Cinv_uvh.y, Cinv_uvh.z) < 1E-12 || D ==0)
    {
        normal = {0, 0, 0};
        ray_plane = {0, 0};
    }
    else
    {
        T l = length(tx, ty, z);
        mat3<T> nJ_T = glm::mat3(rz, 0.f, -tx * rz2,             // 1st column
                                0.f, rz, -ty * rz2,              // 2nd column
                                tx/l, ty/l, z/l                  // 3rd column
        );
        T uu = u * u;
        T vv = v * v;
        T uv = u * v;

        mat3x2<T> nJ_inv_T = mat3x2<T>(vv + 1, -uv,          // 1st column
                                       -uv, uu + 1,          // 2nd column
                                        -u, -v               // 3nd column
        );
        T factor = l / (uu + vv + 1);
        vec3<T> Cinv_uvh_n = glm::normalize(Cinv_uvh);
        T u_Cinv_u_n_clmap = max(glm::dot(Cinv_uvh_n, uvh), 1E-7);
        vec2<T> plane = nJ_inv_T * (Cinv_uvh_n / u_Cinv_u_n_clmap);
        vec3<T> ray_normal_vector = {-plane.x*factor, -plane.y*factor, -1};
		vec3<T> cam_normal_vector = nJ_T * ray_normal_vector;
		normal = glm::normalize(cam_normal_vector);
        ray_plane = {plane.x * factor / fx, plane.y * factor / fy};

        if constexpr (INTE)
        {
            glm::mat3 inv_cov_ray;
            if (well_conditioned) 
            {
                float ltz = uu + vv + 1;

                glm::mat3 nJ_inv_full = z / (uu + vv + 1) * \
                                        glm::mat3(
                                            uu+1, -uv, u/l*ltz,
                                            -uv, uu+1, v/l*ltz,
                                            -u, -v, 1/l*ltz);
                glm::mat3 T2 = W * glm::transpose(nJ_inv_full);
                inv_cov_ray = glm::transpose(T2) * cov3d_inv * T2;
            }
            else 
            {
                glm::mat3 T2 = W * nJ_T;
                glm::mat3 cov_ray = glm::transpose(T2) * cov3d_inv * T2;
                glm::mat3 cov_eigen_vector;
                glm::vec3 cov_eigen_value;
                glm_modification::findEigenvaluesSymReal(cov_ray, cov_eigen_value, cov_eigen_vector);
                unsigned int min_id = cov_eigen_value[0]>cov_eigen_value[1]? (cov_eigen_value[1]>cov_eigen_value[2]?2:1):(cov_eigen_value[0]>cov_eigen_value[2]?2:0);
				float lambda1 = cov_eigen_value[(min_id+1)%3];
				float lambda2 = cov_eigen_value[(min_id+2)%3];
				float lambda3 = cov_eigen_value[min_id];
				glm::mat3 new_cov_eigen_vector = glm::mat3();
				new_cov_eigen_vector[0] = cov_eigen_vector[(min_id+1)%3];
				new_cov_eigen_vector[1] = cov_eigen_vector[(min_id+2)%3];
				new_cov_eigen_vector[2] = cov_eigen_vector[min_id];
				glm::vec3 r3 = glm::vec3(new_cov_eigen_vector[0][2],new_cov_eigen_vector[1][2],new_cov_eigen_vector[2][2]);

				glm::mat3 cov2d = glm::mat3(
					1/lambda1,0,-r3[0]/r3[2]/lambda1,
					0,1/lambda2,-r3[1]/r3[2]/lambda2,
					-r3[0]/r3[2]/lambda1,-r3[1]/r3[2]/lambda2,0
				);
				glm::mat3 inv_cov_ray = new_cov_eigen_vector * cov2d * glm::transpose(new_cov_eigen_vector);
            }
			glm::mat3 scale = glm::mat3(1/fx,0,0,
										0, 1/fy,0,
										0,0,1);
			inv_cov_ray = scale * inv_cov_ray * scale;

            invraycov3Ds[0] = inv_cov_ray[0][0];
            invraycov3Ds[1] = inv_cov_ray[0][1];
            invraycov3Ds[2] = inv_cov_ray[0][2];
            invraycov3Ds[3] = inv_cov_ray[1][1];
            invraycov3Ds[4] = inv_cov_ray[1][2];
            invraycov3Ds[5] = inv_cov_ray[2][2];
        }
    }

    return well_conditioned;
}


template <typename T>
inline __device__ void rade_persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d, const mat3<T> cov3d, const T fx, const T fy, const T cx,
    const T cy, const uint32_t width, const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d, const vec2<T> v_mean2d, const vec2<T> v_ray_plane, const vec3<T> v_normal,
    // grad inputs
    vec3<T> &v_mean3d, mat3<T> &v_cov3d) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x = 1.3f * tan_fovx;
    T lim_y = 1.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T u = min(lim_x, max(-lim_x, x * rz));
    T v = min(lim_y, max(-lim_y, y * rz));
    T tx = z * u;
    T ty = z * v;
    mat3<T> v_cov3d_ = {0,0,0,0,0,0,0,0,0};

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(fx * rz, 0.f,                  // 1st column
                            0.f, fy * rz,                  // 2nd column
                            -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );

    // calculate the ray space intersection plane.
    auto length = [](T x, T y, T z) { return sqrt(x*x+y*y+z*z); };
    mat3<T> cov3d_eigen_vector;
	vec3<T> cov3d_eigen_value;
	int D = glm_modification::findEigenvaluesSymReal(cov3d,cov3d_eigen_value,cov3d_eigen_vector);
    unsigned int min_id = cov3d_eigen_value[0]>cov3d_eigen_value[1]? (cov3d_eigen_value[1]>cov3d_eigen_value[2]?2:1):(cov3d_eigen_value[0]>cov3d_eigen_value[2]?2:0);
    mat3<T> cov3d_inv;
	bool well_conditioned = cov3d_eigen_value[min_id]>1E-8;
	vec3<T> eigenvector_min;
	if(well_conditioned)
	{
		mat3<T> diag = mat3<T>( 1/cov3d_eigen_value[0], 0, 0,
                                0, 1/cov3d_eigen_value[1], 0,
                                0, 0, 1/cov3d_eigen_value[2] );
		cov3d_inv = cov3d_eigen_vector * diag * glm::transpose(cov3d_eigen_vector);
	}
	else
	{
		eigenvector_min = cov3d_eigen_vector[min_id];
		cov3d_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
    vec3<T> uvh = {u, v, 1};
    vec3<T> Cinv_uvh = cov3d_inv * uvh;
    T l, v_u, v_v, v_l;
    mat3<T> v_nJ_T;
    if(length(Cinv_uvh.x, Cinv_uvh.y, Cinv_uvh.z) < 1E-12 || D ==0)
    {
        l = 1.f;
        v_u = 0.f;
        v_v = 0.f;
        v_l = 0.f;
        v_nJ_T = {0,0,0,0,0,0,0,0,0};
    }
    else
    {
        l = length(tx, ty, z);
        mat3<T> nJ_T = glm::mat3(rz, 0.f, -tx * rz2,             // 1st column
                                0.f, rz, -ty * rz2,              // 2nd column
                                tx/l, ty/l, z/l                  // 3rd column
        );
        T uu = u * u;
        T vv = v * v;
        T uv = u * v;

        mat3x2<T> nJ_inv_T = mat3x2<T>(vv + 1, -uv,          // 1st column
                                       -uv, uu + 1,          // 2nd column
                                        -u, -v               // 3nd column
        );
        const T nl = uu + vv + 1;
        T factor = l / nl;
        vec3<T> Cinv_uvh_n = glm::normalize(Cinv_uvh);
        T u_Cinv_u = glm::dot(Cinv_uvh, uvh);
        T u_Cinv_u_n = glm::dot(Cinv_uvh_n, uvh);
        T u_Cinv_u_clmap = max(u_Cinv_u, 1E-7);
        T u_Cinv_u_n_clmap = max(u_Cinv_u_n, 1E-7);
        mat3<T> cov3d_inv_u_Cinv_u = cov3d_inv / u_Cinv_u_clmap;
        vec3<T> Cinv_uvh_u_Cinv_u = Cinv_uvh_n / u_Cinv_u_n_clmap;
        vec2<T> plane = nJ_inv_T * Cinv_uvh_u_Cinv_u;
        vec3<T> ray_normal_vector = {-plane.x*factor, -plane.y*factor, -1};
		vec3<T> cam_normal_vector = nJ_T * ray_normal_vector;
		vec3<T> normal = glm::normalize(cam_normal_vector);
        // vec2<T> ray_plane = {plane.x * factor / fx, plane.y * factor / fy};

        T cam_normal_vector_length = glm::length(cam_normal_vector);

        vec3<T> v_normal_l = v_normal / cam_normal_vector_length;
        vec3<T> v_cam_normal_vector = v_normal_l - normal * glm::dot(normal,v_normal_l);
        vec3<T> v_ray_normal_vector = glm::transpose(nJ_T) * v_cam_normal_vector;
        v_nJ_T = glm::outerProduct(v_cam_normal_vector, ray_normal_vector);

        vec2<T> ray_plane_uv = {plane.x * factor, plane.y * factor};
        const vec2<T> v_ray_plane_uv = {v_ray_plane.x / fx, v_ray_plane.y / fy};
        v_l = glm::dot(plane, -glm::make_vec2(v_ray_normal_vector) + v_ray_plane_uv) / nl;
        vec2<T> v_plane = {factor * (-v_ray_normal_vector.x + v_ray_plane_uv.x),
                            factor * (-v_ray_normal_vector.y + v_ray_plane_uv.y)};
        T v_nl = (-v_ray_normal_vector.x * ray_normal_vector.x - v_ray_normal_vector.y * ray_normal_vector.y
                    -v_ray_plane.x*ray_plane_uv.x - v_ray_plane.y*ray_plane_uv.y) / nl;
        
        T tmp = glm::dot(v_plane, plane);
        if(well_conditioned)
        {
            v_cov3d_ += -glm::outerProduct(Cinv_uvh,
                            cov3d_inv_u_Cinv_u * (uvh * (-tmp) + glm::transpose(nJ_inv_T) * v_plane));
        }
        else
        {
            mat3<T> v_cov3d_inv = glm::outerProduct(uvh,
                                    (-tmp * uvh + glm::transpose(nJ_inv_T) * v_plane) / u_Cinv_u_clmap);
            vec3<T> v_eigenvector_min = (v_cov3d_inv + glm::transpose(v_cov3d_inv)) * eigenvector_min;
            for(int j =0;j<3;j++)
			{
				if(j!=min_id)
				{
					T scale = glm::dot(cov3d_eigen_vector[j], v_eigenvector_min)/min(cov3d_eigen_value[min_id] - cov3d_eigen_value[j], - 0.0000001f);
					v_cov3d_ += glm::outerProduct(cov3d_eigen_vector[j] * scale, eigenvector_min);
				}
			}
        }

        vec3<T> v_uvh = cov3d_inv_u_Cinv_u * (2 * (-tmp) * uvh +  glm::transpose(nJ_inv_T) * v_plane);
        mat3x2<T> v_nJ_inv_T = glm::outerProduct(v_plane, Cinv_uvh_u_Cinv_u);

        // derivative of u v in factor, uvh and nJ_inv_T variables.
        v_u = v_nl * 2 * u //dnl/du
            + v_uvh.x
            + (v_nJ_inv_T[0][1] + v_nJ_inv_T[1][0]) * (-v) + 2 * v_nJ_inv_T[1][1] * u - v_nJ_inv_T[2][0];
        v_v = v_nl * 2 * v //dnl/du
            + v_uvh.y
            + (v_nJ_inv_T[0][1] + v_nJ_inv_T[1][0]) * (-u) + 2 * v_nJ_inv_T[0][0] * v - v_nJ_inv_T[2][1];

    }

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d_ += glm::transpose(J) * v_cov2d * J;

    v_cov3d += v_cov3d_;

    vec3<T> v_mean3d_ = {0,0,0};
    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d_ += vec3<T>(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    T l3 = l * l * l;
    T v_mean3d_x = -fx * rz2 * v_J[2][0] + v_u * rz
                    - v_nJ_T[0][2]*rz2 + v_nJ_T[2][0]*(1/l-tx*tx/l3) + (v_nJ_T[2][1] * ty + v_nJ_T[2][2] * z)*(-tx/l3)
                    + v_l * tx / l;
    T v_mean3d_y = -fy * rz2 * v_J[2][1]  + v_v * rz
                    - v_nJ_T[1][2]*rz2 + (v_nJ_T[2][0]* tx + v_nJ_T[2][2]* z) *(-ty/l3) + v_nJ_T[2][1]*(1/l-ty*ty/l3)
                    + v_l * ty / l;
    if (x * rz <= lim_x && x * rz >= -lim_x) {
        v_mean3d_.x += v_mean3d_x;
    } else {
        // v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
        v_mean3d_.z += v_mean3d_x * u;
    }
    if (y * rz <= lim_y && y * rz >= -lim_y) {
        v_mean3d_.y += v_mean3d_y;
    } else {
        // v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
        v_mean3d_.z += v_mean3d_y * v;
    }
    v_mean3d_.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1]
                  + 2.f * fx * tx * rz3 * v_J[2][0] + 2.f * fy * ty * rz3 * v_J[2][1]
                  - (v_u * tx + v_v * ty) * rz2
			      + v_nJ_T[0][0] * (-rz2) + v_nJ_T[1][1] * (-rz2) + v_nJ_T[0][2] * (2 * tx * rz3) + v_nJ_T[1][2] * (2 * ty * rz3)
				  + (v_nJ_T[2][0] * tx + v_nJ_T[2][1] * ty) * (-z/l3) + v_nJ_T[2][2] * (1 / l - z * z / l3)
				  + v_l * z / l;
    v_mean3d += v_mean3d_;
}

//

} // namespace gsplat

#endif // GSPLAT_CUDA_UTILS_H
