import numpy as np
import scipy.integrate as integrate
import time
import pandas as pd
import jax
import jax.numpy as jnp
import argparse

# Verify that JAX is detecting the GPU
print(f"Available JAX devices: {jax.devices()}")

TANH_NORM_CONST = 0.6279

rng = np.random.default_rng(42)

# Functions

def initialize_student_weights(norm_type, C_matrix, N_dim, random_generator):
    """Initialize student weights distributed on the chosen manifold."""
    W = random_generator.standard_normal(N_dim) # Initialize the W (student W), N-dimensional vectors with components W_i≈N(0,1)
    if norm_type == 'sphere':
        W = W * np.sqrt(N_dim) / np.linalg.norm(W) # W uniformly distributed on the N-sphere of radius sqrt(N)
    elif norm_type == 'ellipse':
        proj_norm = (W.T @ C_matrix @ W) / N_dim # Normalization on the ellipse
        W = W / np.sqrt(proj_norm)
    return W

def loss_fn(W, sigmaFx, Ystar_sq, denominator, P_val):
    """Calculate the loss scalar."""
    N_dim = W.shape[0]
    Y = (sigmaFx @ W) / jnp.sqrt(N_dim) # Compute the student labels Y
    Y_sq = Y**2 # Y^2
    numerator = (Ystar_sq - Y_sq)**2
    return jnp.sum(numerator / denominator) / P_val #loss

def build_train_step(norm_type):
    """
    Builds the step function for the lax.scan, specialized on the norm_type.
    """
    loss_grad_fn = jax.value_and_grad(loss_fn)

    def train_step(carry, epoch_idx):
        W_current, sigmaFx, Ystar_sq, denominator, C, FWstar, norm_FWstar, P_val, eta, F_mat, Wstar_mat, norm_Wstar = carry

        N_dim = W_current.shape[0]

        # Compute Gradient
        loss, grads = loss_grad_fn(W_current, sigmaFx, Ystar_sq, denominator, P_val)

        # Update the weights based on the manifold
        if norm_type == 'ellipse':
            v = C @ W_current # Vector normal to the ellipse
            grads = grads - (jnp.dot(v, grads) / jnp.dot(v, v)) * v
            W_tilde = W_current - eta * grads
            proj_norm = (W_tilde.T @ C @ W_tilde) / N_dim
            W_next = W_tilde / jnp.sqrt(proj_norm) # Renormalize on the elliptic manifold
            FW = F_mat @ W_next
            overlap = jnp.abs((Wstar_mat @ FW) / (norm_Wstar * jnp.linalg.norm(FW))) # Elliptic normalization overlap

        elif norm_type == 'sphere':
            W_tilde = W_current - eta * grads
            W_next = W_tilde * jnp.sqrt(N_dim) / jnp.linalg.norm(W_tilde) # Renormalize on the spherical manifold
            overlap = jnp.abs((W_next @ FWstar) / (jnp.linalg.norm(W_next) * norm_FWstar)) # Spherical normalization overlap

        # Pack updated state and constant arrays into the carry tuple for the next iteration
        next_carry = (W_next, sigmaFx, Ystar_sq, denominator, C, FWstar, norm_FWstar, P_val, eta, F_mat, Wstar_mat, norm_Wstar)
        return next_carry, (overlap, loss)

    return train_step


def build_run_training(norm_type, epochs):
    train_step = build_train_step(norm_type)
    checkpoint_1 = int(epochs**(5 / 6))
    checkpoint_2 = int(epochs**(11 / 12))

    @jax.jit
    def run_training(W_start, sigmaFx, Ystar_sq, denominator, C, FWstar, norm_FWstar, P_val, eta, F_mat, Wstar_mat, norm_Wstar):
        initial_carry = (W_start, sigmaFx, Ystar_sq, denominator, C, FWstar, norm_FWstar, P_val, eta, F_mat, Wstar_mat, norm_Wstar)
        final_carry, (ov_history, loss_history) = jax.lax.scan(
            train_step, initial_carry, xs=None, length=epochs
        )
        W_final = final_carry[0]
        return W_final, ov_history[-1], loss_history[-1], ov_history[checkpoint_1], ov_history[checkpoint_2]

    return run_training

# Main

def main():
  # CSV
  csv_data = []

  S = 5  # Number of samples for each (alpha, alphaD) point

  # Parameters

  parser = argparse.ArgumentParser(description="JAX Student-Teacher GD Simulation")
  parser.add_argument("--N", type=int, default=1000, help="Student width")
  parser.add_argument("--epochs", type=int, default=1000000, help="GD epochs, 100000 for testing, 1000000 for the actual simulations")
  parser.add_argument("--eta", type=float, default=0.2, help="Learning rate")
  parser.add_argument("--a", type=float, default=0.01, help="Loss function parameter")
  parser.add_argument("--norm", type=str, default="ellipse", help="Normalization: 'sphere' or 'ellipse'")
  parser.add_argument("--sig", type=str, default="yes", help="Signal: 'yes', 'no'")
  args = parser.parse_args()

  # The (alpha, alphaD) point on which to iterate
  alphaD_fixed = 0.5
  alphas = np.concatenate([
      np.array([0.01]),
      np.arange(1, 6, 1),
  ])
  test_points = [(alpha, alphaD_fixed) for alpha in alphas]

  # Analytical constants for the chosen activation function (tanh).
  c = TANH_NORM_CONST
  sigma_func = lambda z: np.tanh(z) / c
  gaussian = lambda z: np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

  kappa1, _ = integrate.quad(lambda z: z * sigma_func(z) * gaussian(z), -np.inf, np.inf)
  kappa_norm, _ = integrate.quad(lambda z: (sigma_func(z)**2) * gaussian(z), -np.inf, np.inf)
  kappa_star = np.sqrt(kappa_norm - kappa1**2)

  run_training_jit = build_run_training(args.norm, args.epochs)

  for alpha, alphaD in test_points:

      print("\n" + "="*60)
      print(f"--- alpha = {alpha}, alphaD = {alphaD} ---")
      print("="*60)

      D_val = int(alphaD * args.N)
      P_val = int(alpha * args.N)
      eta = args.eta
      for s in range(S):
          msg = f" -> Sample {s+1}/{S} for point ({alpha}, {alphaD})..."
          print(f"{msg}", end=" ", flush=True)

          # Initialization
          Wstar = rng.standard_normal(D_val) # Make Wstar (teacher W), D-dimensional vector with components Wstar_i≈N(0,1)
          Wstar = Wstar * np.sqrt(D_val) / np.linalg.norm(Wstar) # Ensure that Wstar is uniformly distributed on the D-sphere of radius sqrt(D)
          F = rng.standard_normal((D_val, args.N)) # Make the fixed F matrix, with size DxN and with components Fij≈N(0,1)
          X = rng.standard_normal((P_val, D_val)) / np.sqrt(D_val) # Make the inputs matrix
                                                                  # Since there are P examples, each of the P lines is a D-dimensional input vector
                                                                  # Each component is Xij≈N(0,1/D)
          # Make the teacher outputs
          if args.sig=='yes':
            Ystar = X @ Wstar
          elif args.sig=='no':
            Ystar = rng.standard_normal(P_val)

          Fx = X @ F # Overparametrization of the examples through the F matrix. From D-dimensional vectors to N-dimensional vectors
          sigmaFx = np.tanh(Fx)/TANH_NORM_CONST # Activation function applied to Fx

          C_matrix = (kappa1**2 * (F.T @ F) / D_val) + (kappa_star**2 * np.eye(args.N)) # C matrix to take into account the non_linearity

          W = initialize_student_weights(args.norm, C_matrix, args.N, rng)

          FWstar = Wstar @ F # Overparametrization of the teacher Wstar from D-dimensional to N-dimensional.
          norm_FWstar = np.linalg.norm(FWstar) # Norm of FWstar
          Ystar_sq = Ystar**2 # Part of the Hessian matrix
          denominator = 2*(args.a + Ystar_sq) # Part of the Hessian matrix

          # Convert to JAX arrays
          W_init_j = jnp.array(W)
          sigmaFx_j = jnp.array(sigmaFx)
          Ystar_sq_j = jnp.array(Ystar_sq)
          denominator_j = jnp.array(denominator)
          FWstar_j = jnp.array(FWstar)
          norm_FWstar_j = jnp.array(norm_FWstar)
          C_j = jnp.array(C_matrix)
          P_val_j = jnp.array(P_val)
          F_j = jnp.array(F)
          Wstar_j = jnp.array(Wstar)
          norm_Wstar_j = jnp.linalg.norm(Wstar_j)


          start_time = time.time()
          # Gradient Descent

          W = W_init_j
          delta1 = 1
          delta2 = 1
          final_loss = 10
          trial = 0
          while delta1 > 1e-4 or delta2 > 1e-4 or final_loss > 2:
            eta_j = jnp.array(eta)
            W, final_ov, final_loss, mid1_ov, mid2_ov = run_training_jit(
                W, sigmaFx_j, Ystar_sq_j, denominator_j, C_j, FWstar_j, norm_FWstar_j, P_val_j, eta_j, F_j, Wstar_j, norm_Wstar_j
            )
            final_ov = float(final_ov)
            final_loss = float(final_loss)
            delta1 = abs(float(mid1_ov - final_ov))
            delta2 = abs(float(mid2_ov - final_ov))
            if trial > 0:
              print(" " * len(msg), end=" ", flush=True)
            if final_loss > 20:
              print(f"Completed in {time.time() - start_time:6.2f}s | Final Overlap: {final_ov:.4f} | Final Loss: {final_loss:.0f} | eta = {eta} | Delta1 = {delta1:.4f}, Delta2 = {delta2:.4f}, Loss > 20, try with smaller eta...")
              eta = eta/10
              W = jnp.array(initialize_student_weights(args.norm, C_matrix, args.N, rng))
            elif delta1 > 1e-4 or delta2 > 1e-4:
              print(f"Completed in {time.time() - start_time:6.2f}s | Final Overlap: {final_ov:.4f} | Final Loss: {final_loss:.4f} | eta = {eta} | Delta1 = {delta1:.4f}, Delta2 = {delta2:.4f}, Delta > 0, continue...")
            else:
              print(f"Completed in {time.time() - start_time:6.2f}s | Final Overlap: {final_ov:.4f} | Final Loss: {final_loss:.4f} | eta = {eta} | Delta1 = {delta1:.4f}, Delta2 = {delta2:.4f}")
            trial += 1
          # Save data
          csv_data.append({
              'alpha': alpha,
              'alphaD': alphaD,
              'sample': s + 1,
              'final_gd_overlap': final_ov
          })

  # CSV
  print("\n" + "="*60)
  print("                    Final Global Results")
  print("="*60)

  df_results = pd.DataFrame(csv_data)

  # Show a preview of the first results
  print("\nFirst lines of the dataset:")
  print(df_results.head(10).to_string(index=False))

  print("\nAverages grouped by alpha:")
  recap = df_results.groupby(['alpha', 'alphaD']).mean().reset_index()
  # Remove the sample column
  recap = recap.drop(columns=['sample'])
  print(recap.to_string(index=False))

  # Save on the CSV
  csv_filename = f"gd_final_overlap_N{args.N}_alphaD{alphaD}_{args.norm}_a{args.a}_{S}sample.csv"
  df_results.to_csv(csv_filename, index=False)
  print(f"\nAll the data has been successfully saved in: {csv_filename}")

if __name__ == "__main__":
  main()
