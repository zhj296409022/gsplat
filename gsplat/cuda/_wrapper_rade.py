from typing import Callable, Optional, Tuple, Literal

import torch
from torch import Tensor

from ._wrapper import _make_lazy_cuda_func


def fully_fused_projection_integration(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionIntegrationPacked.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
            camera_model,
        )
    else:
        return _FullyFusedProjectionIntegration.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
        )


def integrate_to_points(
    points2d: Tensor,  # [C, N, 2] or [nnz, 2]
    point_depths: Tensor,  # [C, N] or [nnz]
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    view2guassian: Tensor,  # [C, N, 10] or [nnz, 10]
    ray_planes: Tensor, # [C, N, 2] or [nnz, 2]
    ray_ts: Tensor,  # [C, N] or [nnz]
    invraycov3Ds: Tensor, # [C, N, 6] or [nnz, 6]
    conditions: Tensor,  # [C, N] or [nnz]
    Ks: Tensor,  # [C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    point_isect_offsets: Tensor,  # [C, tile_height, tile_width]
    point_flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """
    assert not packed, "integrate_to_points only supports non-packed mode."
    
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
        assert view2guassian.shape == (nnz, 10), view2guassian.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
        assert view2guassian.shape == (C, N, 10), view2guassian.shape
        assert Ks.shape == (C, 3, 3), Ks.shape
        PN = points2d.size(1)
        assert points2d.shape == (C, PN, 2), points2d.shape
        assert point_depths.shape == (C, PN), point_depths.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    return_colors, render_alphas, render_colors, point_coordinates, point_sdf = _make_lazy_cuda_func(
            "rade_integrate_to_points_fwd"
        )(
            points2d,
            point_depths,
            means2d,
            conics,
            colors,
            opacities,
            view2guassian,
            ray_planes,
            ray_ts,
            invraycov3Ds,
            conditions,
            Ks,
            backgrounds,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
            point_isect_offsets,
            point_flatten_ids,
        )

    return return_colors, render_alphas, render_colors, point_coordinates, point_sdf


def fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and perspective projection (:func:`persp_proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
        )
    else:
        return _FullyFusedProjection.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
      
        
def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    ray_ts: Tensor, # [C, N] or [nnz]
    ray_planes: Tensor,  # [C, N, 2] or [nnz, 2]
    normals: Tensor,  # [C, N, 3] or [nnz, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    Ks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    require_geo: bool = False,
    record_transmittance: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
        if require_geo:
            assert ray_ts.shape == nnz, means2d.shape
            assert ray_planes.shape == (nnz, 2), conics.shape
            assert normals.shape == (nnz, 3), colors.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
        if require_geo:
            assert ray_ts.shape == (C, N), means2d.shape
            assert ray_planes.shape == (C, N, 2), conics.shape
            assert normals.shape == (C, N, 3), colors.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"
    transmittance = None
    if require_geo:
        render_colors, render_alphas, expected_depths, median_depths, expected_normals = _RasterizeToPixels_wDepth.apply(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            ray_ts.contiguous(),
            ray_planes.contiguous(),
            normals.contiguous(),
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            Ks.contiguous(),
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            absgrad,
        )
    else:
        render_colors, render_alphas, transmittance = _RasterizeToPixels.apply(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            absgrad,
            record_transmittance
        )
        expected_depths, median_depths, expected_normals = None, None, None

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas, expected_depths, median_depths, expected_normals, transmittance


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals = _make_lazy_cuda_func(
            "rade_fully_fused_projection_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations, v_ray_ts, v_ray_planes, v_normals):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "rade_fully_fused_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_ray_ts.contiguous(),
            v_ray_planes.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        record_transmittance: bool
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids, transmittance = _make_lazy_cuda_func(
            "rade_rasterize_to_pixels_wo_depth_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            record_transmittance
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas, transmittance.detach() if transmittance is not None else transmittance

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        _: Tensor
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rade_rasterize_to_pixels_wo_depth_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    
class _RasterizeToPixels_wDepth(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        ray_ts: Tensor,  # [C, N]
        ray_planes: Tensor,  # [C, N, 2]
        normals: Tensor,  # [C, N, 3]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        Ks: Tensor, # [C, 3, 3]
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, expected_depths, median_depths, expected_normals, median_ids, last_ids = _make_lazy_cuda_func(
            "rade_rasterize_to_pixels_w_depth_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            ray_ts,
            ray_planes,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            Ks,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            ray_ts,
            ray_planes,
            normals,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            median_ids,
            Ks
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas, expected_depths, median_depths, expected_normals

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_render_expected_depths: Tensor,  # [C, H, W, 1]
        v_render_median_depths: Tensor,  # [C, H, W, 1]
        v_render_expected_normals: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            ray_ts,
            ray_planes,
            normals,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            median_ids,
            Ks
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        # print(means2d.dtype,conics.dtype,colors.dtype,opacities.dtype,ray_ts.dtype,ray_planes.dtype,normals.dtype,v_render_expected_depths.dtype,v_render_expected_normals.dtype)
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_ray_ts,
            v_ray_planes,
            v_normals
        ) = _make_lazy_cuda_func("rade_rasterize_to_pixels_w_depth_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            ray_ts,
            ray_planes,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            Ks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_expected_depths.contiguous(),
            v_render_median_depths.contiguous(),
            v_render_expected_normals.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[7]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_ray_ts,
            v_ray_planes,
            v_normals,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
            ray_ts, 
            ray_planes, 
            normals
        ) = _make_lazy_cuda_func("rade_fully_fused_projection_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad

        return camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals

    @staticmethod
    def backward(
        ctx,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        v_ray_ts, 
        v_ray_planes, 
        v_normals
    ):
        (
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "rade_fully_fused_projection_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_ray_ts.contiguous(), 
            v_ray_planes.contiguous(), 
            v_normals.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_covars,  # [nnz, 6]
                    size=covars.size(),  # [N, 6]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionIntegration(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals, invraycov3Ds, conditions = _make_lazy_cuda_func(
            "rade_fully_fused_projection_integration_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations, invraycov3Ds, conditions
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations, v_ray_ts, v_ray_planes, v_normals, v_invraycov3Ds, v_conditions):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "rade_fully_fused_projection_integration_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_ray_ts.contiguous(),
            v_ray_planes.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionIntegrationPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
            ray_ts, 
            ray_planes, 
            normals,
            invraycov3Ds,
            conditions
        ) = _make_lazy_cuda_func("rade_fully_fused_projection_integration_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad

        return camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals, invraycov3Ds, conditions

    @staticmethod
    def backward(
        ctx,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        v_ray_ts, 
        v_ray_planes, 
        v_normals,
        v_invraycov3Ds,
        v_conditions
    ):
        (
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "rade_fully_fused_projection_integration_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_ray_ts.contiguous(), 
            v_ray_planes.contiguous(), 
            v_normals.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_covars,  # [nnz, 6]
                    size=covars.size(),  # [N, 6]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )