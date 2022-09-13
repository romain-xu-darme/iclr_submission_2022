import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .utils import migrate
from .utils import normalize_min_max, polarity_and_collapse
from progress.bar import Bar
from typing import List, Callable, Optional, Tuple
from PIL import Image


def _get_output_at_locations(
        model: nn.Module,
        img_array: np.array,
        preprocess: Callable,
        locs: List[tuple],
        device: Optional[str] = 'cpu',
) -> List[np.array]:
    """ Given a model, an input tensor, returns the list of values
    at a set of output locations

    :param model: Model
    :param img_array: Input image
    :param preprocess: Preprocessing function
    :param locs: List of output locations
    :param device: Target device
    """
    # Preprocess input
    T = preprocess(img_array)
    if len(T.size()) != 4:
        T = T[None]
    # Map to device
    T = T.to(device, dtype=torch.float, non_blocking=True)

    # Single forward pass
    outputs = migrate(model(T))
    res = []
    for p, h, w in locs:
        res.append(outputs[0, p, h, w])
    return res


def _get_gradients_at_locations(
        model: nn.Module,
        img_array: np.array,
        preprocess: Callable,
        locs: List[tuple],
        device: Optional[str] = 'cpu',
        verbose: Optional[bool] = False,
        prefix: Optional[str] = 'Computing gradients',
) -> List[np.array]:
    """ Given a model, an input tensor, returns the list of gradients
    at a set of output locations

    :param model: Model
    :param img_array: Input image
    :param preprocess: Preprocessing function
    :param locs: List of output locations
    :param device: Target device
    :param verbose: Show progress bar
    :param prefix: Progress bar prefix
    """
    # Preprocess input
    T = preprocess(img_array)
    if len(T.size()) != 4:
        T = T[None]
    # Map to device
    T = T.to(device, dtype=torch.float, non_blocking=True)
    T.requires_grad_()

    # Single forward pass
    outputs = model(T)
    bar = Bar(prefix, max=len(locs), suffix='%(percent).1f%%') if verbose else None
    res = []
    for p, h, w in locs:
        output = outputs[0, p, h, w]
        output.backward(retain_graph=True)
        res.append(migrate(T.grad.data[0]))
        # Reset gradients
        T.grad.data.zero_()
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    return res


def _get_integrated_gradients_at_locations(
        model: nn.Module,
        img_array: np.array,
        preprocess: Callable,
        locs: List[tuple],
        device: Optional[str] = 'cpu',
        nsteps: Optional[int] = 50,
        verbose: Optional[bool] = False,
        prefix: Optional[str] = 'Computing integrated gradients',
) -> List[np.array]:
    """ Given a model, an input tensor, returns the list of integrated gradients
    at a set of output locations

    :param model: Model
    :param img_array: Input image
    :param preprocess: Preprocessing function
    :param locs: List of output locations
    :param device: Target device
    :param nsteps: Number integration steps
    :param verbose: Show progress bar
    :param prefix: Progress bar prefix
    """
    # Create random baseline
    baseline = np.random.rand(*img_array.shape)

    # Disable normalization layer to compute gradients on correlation scores directly
    normalize = model.enable_normalization
    model.enable_normalization = False

    # If input tensor has large values, increase baseline accordingly
    if np.amax(img_array) > 10:
        baseline *= 255

    # Perform interpolation in nsteps
    interpolation = [
        baseline + istep / nsteps * (img_array - baseline) for istep in range(nsteps + 1)]
    # Caution!!! Interpolated images must be rounded to uint8 in order to ensure normalization by
    # torchvision.transforms.ToTensor preprocessing function
    interpolation = np.array(interpolation).astype(np.uint8)

    # Compute gradients
    # grads has shape (nsteps+1) x len(locs) x 3 x H x W
    bar = Bar(prefix, max=interpolation.shape[0], suffix='%(percent).1f%%') if verbose else None
    grads = []
    for x in interpolation:
        grads.append(_get_gradients_at_locations(model, x, preprocess, locs, device))
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    grads = np.array(grads)

    # Restore model behavior
    model.enable_normalization = normalize

    res = []
    coeff = np.moveaxis(img_array - baseline[0], -1, 0)
    for lidx in range(len(locs)):
        # Extract all gradients for a given location
        grad = grads[:, lidx]
        # Trapezoidal approximation
        grad = (grad[:-1] + grad[1:]) / 2.0
        grad = np.mean(grad, axis=0)
        grad *= coeff
        res.append(grad)
    return res


def _get_smooth_gradients_at_locations(
        model: nn.Module,
        img_array: np.array,
        preprocess: Callable,
        locs: List[tuple],
        device: Optional[str] = 'cpu',
        nsamples: Optional[int] = 10,
        noise: Optional[float] = 0.2,
        smoother: Optional[bool] = False,
        verbose: Optional[bool] = False,
        prefix: Optional[str] = 'Computing smooth gradients',
) -> np.array:
    """ Given a model, an input tensor, returns the list of smooth gradients
    at a set of output locations

    :param model: Model
    :param img_array: Input image
    :param preprocess: Preprocessing function
    :param locs: List of output locations
    :param device: Target device
    :param nsamples: Number of samples
    :param noise: Noise level
    :param smoother: Weight each gradient map by local value
    :param verbose: Show progress bar
    :param prefix: Progress bar prefix
    """

    # Disable normalization layer to compute gradients on correlation scores directly
    normalize = model.enable_normalization
    model.enable_normalization = False

    # Compute variance from noise ratio
    sigma = (np.max(img_array) - np.min(img_array)) * noise

    # Generate noisy images around original img_array
    noisy_images = [
        img_array + np.random.normal(loc=0, scale=sigma, size=img_array.shape)
        for _ in range(nsamples)
    ]
    # Caution!!! Noisy images must be rounded to uint8 in order to ensure normalization by
    # torchvision.transforms.ToTensor preprocessing function
    noisy_images = np.array(noisy_images)
    noisy_images = np.clip(noisy_images, 0, 255).astype(np.uint8)

    # Compute gradients
    bar = Bar(prefix, max=noisy_images.shape[0], suffix='%(percent).1f%%') if verbose else None
    grads = []
    for x in noisy_images:
        grads.append(_get_gradients_at_locations(model, x, preprocess, locs, device))
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    # grads has shape (nsamples) x len(locs) x 3 x H x W
    grads = np.array(grads)

    if smoother:
        coeffs = [_get_output_at_locations(model, x, preprocess, locs, device) for x in noisy_images]
        # Expand coeff to nsamples x len(locs) x 1 x 1 x 1
        coeffs = np.expand_dims(np.array(coeffs), (2, 3, 4))
        grads = grads * coeffs

    # Restore model behavior
    model.enable_normalization = normalize

    # Average results
    return np.mean(grads, axis=0)


def _weighted_gradients(
        model: nn.Module,
        img: Image.Image,
        index: List[int],
        preprocess: Callable,
        gradient_fn: Callable,
        device: Optional[str] = 'cpu',
        lazy: Optional[bool] = False,
        threshold: Optional[float] = 0.1,
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = True,
        verbose: Optional[bool] = False,
        prefix: Optional[str] = None,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform visualisation using weighted gradients

    :param model: Model
    :param img: Input image
    :param index: List of pattern indices
    :param preprocess: Preprocessing function
    :param gradient_fn: Gradient function to be called on each interesting activation location
    :param device: Target device
    :param lazy: Lazy evaluation using only max activation
    :param threshold: Minimum activation threshold
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param verbose: Show progress bar
    :param prefix: Progress bar prefix
    :return: List of pattern visualisations, confidence values
    """
    # Preprocess image and map to device
    tensor = img
    if preprocess is not None:
        tensor = preprocess(tensor)
    if tensor.dim() != 4:
        tensor = tensor[None]
    tensor = tensor.to(device, non_blocking=True)

    # Compute activation maps and copy to numpy array
    if model.calibrated:
        model.enable_confidence = True
        amaps, confidence = model(tensor)
        confidence = migrate(confidence)[0, index]
        model.enable_confidence = False
    else:
        amaps, confidence = model(tensor), None
    amaps = migrate(amaps)

    N = amaps.shape[1]
    H = amaps.shape[2]
    W = amaps.shape[3]

    # Find interesting locations
    locs = []
    coeffs = []
    for pidx in index:
        vmax = np.amax(amaps[0, pidx])
        vmin = np.amin(amaps[0, pidx])
        # Uniform activation, ignore map
        if vmin == vmax:
            continue
        for h in range(H):
            for w in range(W):
                output = amaps[0, pidx, h, w]
                # Activation value too low or keep max only
                if output != vmax and (output < threshold or lazy):
                    continue
                # Add location
                locs.append((pidx, h, w))
                coeffs.append(output)
    # Get gradients at interesting locations
    grads = gradient_fn(
        model=model,
        img_array=np.array(img),
        preprocess=preprocess,
        locs=locs,
        device=device,
        verbose=verbose,
        prefix=prefix,
    )

    # Initialize heatmaps
    input_size = tensor.size()[1:]
    res = [np.zeros(input_size) for _ in range(N)]

    # Merge results
    for (p, _, _), coeff, grad in zip(locs, coeffs, grads):
        res[p] += coeff * grad
    res = [res[p] for p in index]

    # Post processing
    # Apply polarity filter and channel average
    res = [polarity_and_collapse(heatmap, polarity=polarity, avg_chan=0) for heatmap in res]
    # Gaussian filter
    if gaussian_ksize:
        res = [gaussian_filter(heatmap, sigma=gaussian_ksize) for heatmap in res]
    # Normalize
    if normalize:
        res = [normalize_min_max(heatmap) for heatmap in res]
    return res, confidence


def saliency(
        model: nn.Module,
        img: Image.Image,
        index: List[int],
        preprocess: Callable,
        device: Optional[str] = 'cpu',
        lazy: Optional[bool] = False,
        threshold: Optional[float] = 0.1,
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = True,
        verbose: Optional[bool] = False,
        **kwargs,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform visualisation using saliency map

    :param model: Model
    :param img: Input image
    :param index: List of pattern indices
    :param preprocess: Preprocessing function
    :param device: Target device
    :param lazy: Lazy evaluation using only max activation
    :param threshold: Minimum activation threshold
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param verbose: Show progress bar
    :return: List of pattern visualisations, confidence values
    """
    return _weighted_gradients(
        model=model,
        img=img,
        index=index,
        preprocess=preprocess,
        gradient_fn=_get_gradients_at_locations,
        device=device,
        lazy=lazy,
        threshold=threshold,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        verbose=verbose,
        prefix='Computing gradients',
    )


def integrated_gradients(
        model: nn.Module,
        img: Image.Image,
        index: List[int],
        preprocess: Callable,
        device: Optional[str] = 'cpu',
        lazy: Optional[bool] = False,
        threshold: Optional[float] = 0.1,
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = True,
        nsteps: Optional[int] = 5,
        nruns: Optional[int] = 2,
        verbose: Optional[bool] = False,
        **kwargs,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform visualisation using integrated gradients

    :param model: Model
    :param img: Input image
    :param index: List of pattern indices
    :param preprocess: Preprocessing function
    :param device: Target device
    :param lazy: Lazy evaluation using only max activation
    :param threshold: Minimum activation threshold
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param nsteps: Number of interpolation steps
    :param nruns: Number of runs
    :param verbose: Show progress bar
    :return: List of pattern visualisations, confidence values
    """

    def get_integrated_gradients_fn(
            nsteps: int,
            nruns: int,
    ) -> Callable:
        """ Returns gradient function to be applied on each interesting location

        :param nsteps: Number of interpolation steps
        :param nruns: Number of runs
        """

        def avg_integrated_gradients(
                model: nn.Module,
                img_array: np.array,
                preprocess: Callable,
                locs: List[tuple],
                device: Optional[str] = 'cpu',
                verbose: Optional[bool] = False,
                prefix: Optional[str] = None,
        ) -> List[np.array]:
            """ Given a model, an input tensor, returns the list of
            averaged integrated gradients at a set of output locations

            :param model: Model
            :param img_array: Input image
            :param preprocess: Preprocessing function
            :param locs: List of output locations
            :param device: Target device
            """
            # Average results over multiple runs
            runs = np.array([
                _get_integrated_gradients_at_locations(
                    model=model,
                    img_array=img_array,
                    preprocess=preprocess,
                    locs=locs,
                    device=device,
                    nsteps=nsteps,
                    verbose=verbose,
                    prefix=f'[Run {irun + 1}/{nruns}] {prefix}',
                ) for irun in range(nruns)])
            return np.abs(np.mean(runs, axis=0))

        return avg_integrated_gradients

    return _weighted_gradients(
        model=model,
        img=img,
        index=index,
        preprocess=preprocess,
        gradient_fn=get_integrated_gradients_fn(nsteps, nruns),
        device=device,
        lazy=lazy,
        threshold=threshold,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        verbose=verbose,
        prefix='Computing integrated gradients',
    )


def smooth_grads(
        model: nn.Module,
        img: Image.Image,
        index: List[int],
        preprocess: Callable,
        device: Optional[str] = 'cpu',
        lazy: Optional[bool] = False,
        threshold: Optional[float] = 0.1,
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = True,
        nsamples: Optional[int] = 10,
        noise: Optional[float] = 0.2,
        smoother: Optional[bool] = False,
        verbose: Optional[bool] = False,
        **kwargs,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform visualisation using SmoothGrad

    :param model: Model
    :param img: Input image
    :param index: List of pattern indices
    :param preprocess: Preprocessing function
    :param device: Target device
    :param lazy: Lazy evaluation using only max activation
    :param threshold: Minimum activation threshold
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param nsamples: Number of samples
    :param noise: Noise level
    :param smoother: Weight each gradient map by local value
    :param verbose: Show progress bar
    :return: List of pattern visualisations, confidence values
    """

    def get_smooth_grads_fn(
            nsamples: int,
            noise: float,
            smoother: bool,
    ) -> Callable:
        def smooth_gradients_at_locations(
                model: nn.Module,
                img_array: np.array,
                preprocess: Callable,
                locs: List[tuple],
                device: Optional[str] = 'cpu',
                verbose: Optional[bool] = False,
                prefix: Optional[str] = 'Computing smooth gradients',
        ) -> np.array:
            return _get_smooth_gradients_at_locations(
                model=model,
                img_array=img_array,
                preprocess=preprocess,
                locs=locs,
                device=device,
                nsamples=nsamples,
                noise=noise,
                smoother=smoother,
                verbose=verbose,
                prefix=prefix)

        return smooth_gradients_at_locations

    return _weighted_gradients(
        model=model,
        img=img,
        index=index,
        preprocess=preprocess,
        gradient_fn=get_smooth_grads_fn(nsamples, noise, smoother),
        device=device,
        lazy=lazy,
        threshold=threshold,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        verbose=verbose,
        prefix='Computing smooth gradients',
    )
