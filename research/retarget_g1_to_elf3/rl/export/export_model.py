"""Export trained PPO policy to ONNX and NPZ formats for deployment."""

import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


class PolicyWrapper(torch.nn.Module):
    """Wrapper to export only the policy network (actor) for inference."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        """Forward pass through policy network only.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            action: Action tensor [batch, action_dim] (deterministic)
        """
        # Get action distribution
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)

        # SB3 clips Box actions during predict(); export the same deterministic contract.
        return torch.clamp(mean_actions, -1.0, 1.0)


def export_to_onnx(model_path: str, output_path: str, obs_dim: int = 385, verbose: bool = True):
    """Export PPO policy to ONNX format.

    Args:
        model_path: Path to trained PPO model (.zip file)
        output_path: Output path for ONNX model
        obs_dim: Observation dimension (default: 385 for ELF3)
        verbose: Print export details

    Returns:
        output_path: Path to exported ONNX model
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Extract policy network
    policy = model.policy

    # Wrap for export (policy only, no value function)
    policy_wrapper = PolicyWrapper(policy)
    policy_wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        policy_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    if verbose:
        # Verify export
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print(f"✓ ONNX export successful")
        print(f"  Input: observation [batch, {obs_dim}]")
        print(f"  Output: action [batch, {model.action_space.shape[0]}]")
        print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path


def export_to_npz(model_path: str, output_path: str, verbose: bool = True):
    """Export PPO policy weights to NPZ format.

    Args:
        model_path: Path to trained PPO model (.zip file)
        output_path: Output path for NPZ file
        verbose: Print export details

    Returns:
        output_path: Path to exported NPZ file
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Extract policy state dict
    policy_state_dict = model.policy.state_dict()

    # Filter for policy (actor) weights only
    policy_weights = {}
    for key, value in policy_state_dict.items():
        # Include MLP extractor actor path and action net
        if 'mlp_extractor.policy_net' in key or 'action_net' in key:
            policy_weights[key] = value.cpu().numpy()

    # Save to NPZ
    print(f"Exporting to NPZ: {output_path}")
    np.savez(output_path, **policy_weights)

    if verbose:
        print(f"✓ NPZ export successful")
        print(f"  Layers exported: {len(policy_weights)}")
        for key, value in policy_weights.items():
            print(f"    {key}: {value.shape}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path


def verify_onnx_export(onnx_path: str, original_model_path: str, obs_dim: int = 385, n_tests: int = 5):
    """Verify ONNX export produces same outputs as original model.

    Args:
        onnx_path: Path to exported ONNX model
        original_model_path: Path to original PPO model
        obs_dim: Observation dimension
        n_tests: Number of random tests to run

    Returns:
        True if all tests pass within tolerance
    """
    import onnxruntime as ort

    # Load both models
    original_model = PPO.load(original_model_path)

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    print(f"\nVerifying ONNX export with {n_tests} random tests...")

    all_passed = True
    max_diff = 0.0

    for i in range(n_tests):
        # Generate random observation
        obs = np.random.randn(1, obs_dim).astype(np.float32)

        # Get original model output
        original_action, _ = original_model.predict(obs, deterministic=True)

        # Get ONNX model output
        ort_inputs = {'observation': obs}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_action = ort_outputs[0][0]

        # Compare
        diff = np.abs(original_action - onnx_action).max()
        max_diff = max(max_diff, diff)

        # Check if within tolerance
        tolerance = 1e-5
        passed = diff < tolerance
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"  {status} Test {i+1}: max diff = {diff:.2e}")

    print(f"\n{'✓ All tests passed' if all_passed else '✗ Some tests failed'}")
    print(f"  Maximum difference: {max_diff:.2e}")

    return all_passed


def export_model(model_path: str, output_dir: str, obs_dim: int = 385, verify: bool = True):
    """Export trained model to both ONNX and NPZ formats.

    Args:
        model_path: Path to trained PPO model (.zip file)
        output_dir: Output directory for exports
        obs_dim: Observation dimension
        verify: Whether to verify ONNX export

    Returns:
        dict: Paths to exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model name
    model_name = Path(model_path).stem

    # Export paths
    onnx_path = str(output_dir / f"{model_name}.onnx")
    npz_path = str(output_dir / f"{model_name}_weights.npz")

    print(f"{'='*60}")
    print(f"Exporting Model: {model_path}")
    print(f"{'='*60}\n")

    # Export to ONNX
    export_to_onnx(model_path, onnx_path, obs_dim, verbose=True)

    # Export to NPZ
    export_to_npz(model_path, npz_path, verbose=True)

    # Verify ONNX export
    if verify:
        verify_onnx_export(onnx_path, model_path, obs_dim, n_tests=5)

    print(f"\n{'='*60}")
    print(f"Export Complete")
    print(f"{'='*60}")
    print(f"ONNX: {onnx_path}")
    print(f"NPZ:  {npz_path}")
    print(f"{'='*60}\n")

    return {
        'onnx': onnx_path,
        'npz': npz_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained PPO policy")
    parser.add_argument("model_path", type=str, help="Path to trained PPO model (.zip)")
    parser.add_argument("--output-dir", type=str, default="exports",
                        help="Output directory (default: exports)")
    parser.add_argument("--obs-dim", type=int, default=385,
                        help="Observation dimension (default: 385)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip ONNX verification")

    args = parser.parse_args()

    export_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        obs_dim=args.obs_dim,
        verify=not args.no_verify
    )
