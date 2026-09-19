#!/usr/bin/env python3
"""
magn_texture_tb_mpi.py - Standalone MPI-parallelized band structure and spin-resolved unfolding solver.

Features:
  - Supports 2D / 3D magnetic textures (skyrmions, domain walls, cycloids)
  - Distributes k-points across MPI processes for scalable eigensolving
  - Performs local-frame spin-resolved band unfolding concurrently across MPI ranks
  - Exports complete numerical arrays (.npz) allowing offline plot regeneration
    and color scheme customization in < 1 second without re-diagonalizing
  - Records detailed benchmark and solver execution timing (.json and .txt)
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Prevent threading oversubscription inside MPI ranks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

from pythtb import w90, tb_model
import tb_spinor
import magn_spinor
import unfold_spin_project


def build_kpath(k_nodes, n_per_segment):
    """Generate k-path given high-symmetry nodes and points per segment."""
    labels = [k[0] for k in k_nodes]
    kpts = np.array([k[1] for k in k_nodes], dtype=float)

    k_list = []
    x_list = []
    tick_positions = [0.0]
    x = 0.0

    for i in range(len(kpts) - 1):
        k0 = kpts[i]
        k1 = kpts[i + 1]
        for j in range(n_per_segment):
            t = j / float(n_per_segment)
            k = (1 - t) * k0 + t * k1
            k_list.append(k)
            x_list.append(x)
            if j < n_per_segment - 1:
                dk = np.linalg.norm((k1 - k0) / n_per_segment)
                x += dk
        tick_positions.append(x)

    k_list.append(kpts[-1])
    x_list.append(x)

    return np.array(k_list), np.array(x_list), labels, np.array(tick_positions)


def compute_total_dos_from_evals(evals, sigma=0.02, n_energy=2000, e_min=None, e_max=None):
    """Compute Gaussian-broadened total density of states from band eigenvalues."""
    emin_data = float(np.min(evals))
    emax_data = float(np.max(evals))
    if e_min is None:
        e_min = emin_data - 5.0 * sigma
    if e_max is None:
        e_max = emax_data + 5.0 * sigma
    energy = np.linspace(e_min, e_max, n_energy)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma * evals.shape[1])
    de = energy[:, None, None] - evals[None, :, :]  # (n_energy, nband, nk)
    dos = norm * np.sum(np.exp(-0.5 * (de / sigma) ** 2), axis=(1, 2))
    return energy, dos


def gaussian_spectral_from_bands(
    E,              # (nb, nk)
    w,              # (nb, nk) scalar spectral weight
    s=None,         # (nb, nk, 3) spin weights; optional
    e_grid=None,
    e_min=None,
    e_max=None,
    nE=1000,
    sigma=0.08
):
    """Evaluate Gaussian-broadened spectral function from band weights."""
    E = np.asarray(E, float)
    w = np.asarray(w, float)
    nb, nk = E.shape

    if e_grid is None:
        if e_min is None:
            e_min = E.min() - 5 * sigma
        if e_max is None:
            e_max = E.max() + 5 * sigma
        e_grid = np.linspace(e_min, e_max, nE)
    else:
        e_grid = np.asarray(e_grid, float)
        nE = e_grid.size

    a = np.zeros((nk, nE), float)
    axyz = np.zeros((nk, nE, 3), float) if s is not None else None

    if s is not None:
        s_abs = np.abs(np.asarray(s, float))

    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)

    for ik in range(nk):
        for jb in range(nb):
            g = norm * np.exp(-0.5 * ((e_grid - E[jb, ik]) / sigma) ** 2)
            a[ik] += w[jb, ik] * g
            if s is not None:
                axyz[ik, :, 0] += s_abs[jb, ik, 0] * g
                axyz[ik, :, 1] += s_abs[jb, ik, 1] * g
                axyz[ik, :, 2] += s_abs[jb, ik, 2] * g

    return e_grid, a, axyz


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI-parallelized tight-binding solver for magnetic textures with spin-resolved unfolding."
    )
    parser.add_argument(
        "--w90-dir",
        type=str,
        default="/pscratch/sd/g/gmoore/Workflow_runs/w90_runs/jval_runs/BiFeO3/02runs/00lowres_tests/00run_R3c_std",
        help="Directory containing Wannier90 hr.dat and centres.xyz files."
    )
    parser.add_argument("--seed-up", type=str, default="wannier90.1", help="Seed name for spin-up Wannier model.")
    parser.add_argument("--seed-dn", type=str, default="wannier90.2", help="Seed name for spin-down Wannier model.")
    parser.add_argument("--fermi-level", type=float, default=4.50820836, help="Fermi level in eV.")
    parser.add_argument("--min-hopping-norm", type=float, default=0.01, help="Min hopping norm threshold.")
    parser.add_argument("--nx", type=int, default=8, help="Supercell expansion along x.")
    parser.add_argument("--ny", type=int, default=8, help="Supercell expansion along y.")
    parser.add_argument("--nz", type=int, default=1, help="Supercell expansion along z.")
    parser.add_argument("--nk-per-segment", type=int, default=32, help="k-points per path segment.")
    parser.add_argument(
        "--texture",
        type=str,
        default="skyrmion",
        choices=["skyrmion", "1d_yz", "1d_xz", "1d_xy", "collinear_x", "collinear_z"],
        help="Magnetic texture type (default: skyrmion)."
    )
    parser.add_argument("--r0", type=float, default=1.0, help="Skyrmion / texture core radius.")
    parser.add_argument("--lx", type=float, default=5.0, help="Spatial box scale parameter l_x for magnetic texture.")
    parser.add_argument(
        "--spin-frame",
        type=str,
        default="local",
        choices=["local", "lab"],
        help="Spin projection frame for RGB plotting: 'local' (local rotating frame) or 'lab' (laboratory frame)."
    )
    parser.add_argument("--save-png", action="store_true", default=True, help="Export publication-quality PNG plots on rank 0.")
    parser.add_argument("--png-dir", type=str, default="./plots", help="Directory where PNG figures are saved.")
    parser.add_argument("--output-data", type=str, default="output/unfolded_band_data_8x8.npz", help=".npz file to save numerical outputs for offline plotting.")
    parser.add_argument("--timing-file", type=str, default="timing_info_8x8_p64.json", help="Path to save timing benchmark JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    t_wall_start = time.perf_counter()

    if rank == 0:
        print("=" * 75)
        print("magn_tb - MPI Magnetic Texture Tight-Binding & Unfolding Solver")
        print("=" * 75)
        print(f"MPI Communicator Size : {size} rank(s)")
        print(f"Wannier90 Directory   : {args.w90_dir}")
        print(f"Supercell Dimensions  : [{args.nx}, {args.ny}, {args.nz}]")
        print(f"Magnetic Texture      : {args.texture} (lx={args.lx}, r0={args.r0})")
        print(f"Fermi Level           : {args.fermi_level:.6f} eV")
        print("=" * 75)

    # 1. Define k-path
    g = np.array([0.0, 0.0, 0.0])
    k_delta = 0.1
    s1 = np.array([-k_delta, 0.5, k_delta], dtype=float)
    s2 = np.array([k_delta, 0.5, -k_delta], dtype=float)

    k_nodes = [
        (r"$S_1$", s1),
        (r"$\Gamma$", g),
        (r"$S_2$", s2),
    ]
    k_vec, x_vec, labels, tick_pos = build_kpath(k_nodes, args.nk_per_segment)
    nk_total = len(k_vec)

    if rank == 0:
        print(f"Generated k-path with {nk_total} total k-points.")

    # 2. Supercell specification
    sc_red_lat = np.diag([args.nx, args.ny, args.nz])

    # 3. Load Wannier90 models
    t0_setup = time.perf_counter()
    if rank == 0:
        print("Loading Wannier90 models...")

    w90_up = w90(args.w90_dir, args.seed_up)
    w90_dn = w90(args.w90_dir, args.seed_dn)

    tb_up = w90_up.model(min_hopping_norm=args.min_hopping_norm)
    tb_dn = w90_dn.model(min_hopping_norm=args.min_hopping_norm)

    # 4. Build spinful collinear model
    tb_spinful = tb_spinor.build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=0.0
    )
    dim_r = tb_spinful._dim_r
    base_norb = tb_spinful._norb

    # 5. Build supercell geometry
    sc_tb, sc_vectors = tb_spinor.make_supercell_fast(
        tb_spinful, sc_red_lat, return_sc_vectors=True, to_home=True
    )
    lat_sc = sc_tb.get_lat()
    orb_sc = sc_tb.get_orb()

    # 6. Select magnetic texture
    l_x = args.lx
    # --------------------------------------------------------------------------
    # 1D cycloid / domain wall texture initialization (commented out):
    # if args.texture == "1d_yz":
    #     f_x = lambda x: x / l_x
    #     mag_fn = lambda x, y, z=0.0, r0=1.0: magn_spinor.magnetization_texture_1d(
    #         x, y=y, z=z, r0=r0, f_fn=f_x, plane="yz"
    #     )
    # elif args.texture == "1d_xz":
    #     f_x = lambda x: x / l_x
    #     mag_fn = lambda x, y, z=0.0, r0=1.0: magn_spinor.magnetization_texture_1d(
    #         x, y=y, z=z, r0=r0, f_fn=f_x, plane="xz"
    #     )
    # elif args.texture == "1d_xy":
    #     f_x = lambda x: x / l_x
    #     mag_fn = lambda x, y, z=0.0, r0=1.0: magn_spinor.magnetization_texture_1d(
    #         x, y=y, z=z, r0=r0, f_fn=f_x, plane="xy"
    #     )
    # --------------------------------------------------------------------------
    # Skyrmion texture initialization (commented out):
    # if args.texture == "skyrmion":
    #     mag_fn = magn_spinor.magnetization_texture_skyrmion_00
    # --------------------------------------------------------------------------
    # Constant magnetization everywhere m_z=1 everywhere (see SkyrmionTB nb):
    mag_fn = lambda x, y, z=0.0, r0=1.0: magn_spinor.magnetization_collinear(x, y, z=z, r0=r0, vec=[0.0, 0.0, 1.0])

    # 7. Construct SU(2) spinor rotation samples u_i per Wannier center
    sref = np.array([0.0, 0.0, 1.0], dtype=float)
    u_samples = {}
    x_center, y_center = (0.0, 0.0)

    for sc_i, cell_r in enumerate(sc_vectors):
        cell_key = tuple(int(x) for x in np.asarray(cell_r, dtype=int))
        for wf_i in range(base_norb):
            orb_i = sc_i * base_norb + wf_i
            r_cart = np.dot(orb_sc[orb_i], lat_sc)
            x_cart = float(r_cart[0])
            y_cart = float(r_cart[1])
            z_cart = float(r_cart[2]) if dim_r >= 3 else 0.0

            if args.texture == "skyrmion":
                # Centered on the supercell, scaled to [-0.5*l_x, 0.5*l_x]
                x_tex = (orb_sc[orb_i][0] - 0.5) * l_x
                y_tex = (orb_sc[orb_i][1] - 0.5) * l_x
                n_loc = np.squeeze(mag_fn(x_tex, y_tex, z=z_cart, r0=args.r0))
            else:
                x_tex = (x_cart - x_center)
                y_tex = (y_cart - y_center)
                n_loc = np.squeeze(mag_fn(x_tex, y_tex, z=z_cart, r0=args.r0))

            u_loc = magn_spinor.su2_from_ref_to_n(sref=sref, n=n_loc)
            u_samples[(cell_key, int(wf_i))] = u_loc

    # Reciprocal vectors in supercell coordinates: k_sc = sc_red_lat^T @ k_prim
    k_sc_input = np.array([np.dot(sc_red_lat.T, kpt) for kpt in k_vec], dtype=float)
    t_setup = time.perf_counter() - t0_setup

    if rank == 0:
        print(f"Supercell and texture setup completed in {t_setup:.2f} s.")

    # 8. Partition k-points across MPI processes
    k_indices_all = np.array_split(np.arange(nk_total), size)
    local_indices = k_indices_all[rank]
    n_local = len(local_indices)

    # Pre-assemble rotated supercell TB arrays for fast Numba evaluation
    tb_rot = tb_spinor.apply_local_spin_rotation_from_cellwf_samples(
        tb_spinful, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=True
    )
    onsite_flat, hop_i, hop_j, hop_amps, r_arr, nband_sc = tb_spinor._prepare_tb_arrays_for_numba(tb_rot)

    # 9. Solve and unfold assigned k-points in parallel
    if comm is not None:
        comm.Barrier()
    t0_solve = time.perf_counter()

    evals_local = np.empty((nband_sc, n_local), dtype=np.float64)
    w_pick_local = np.empty((nband_sc, n_local), dtype=np.float64)
    s_lab_local = np.empty((nband_sc, n_local, 3), dtype=np.float64)
    s_loc_local = np.empty((nband_sc, n_local, 3), dtype=np.float64)

    if n_local > 0:
        k_local_sc = k_sc_input[local_indices]
        phases_local = tb_spinor._phases_from_k_and_r(k_local_sc, r_arr)
        h_all_local = tb_spinor._assemble_h_all_k_numba(
            phases_local, onsite_flat, hop_i, hop_j, hop_amps, nband_sc
        )

        for ik_loc in range(n_local):
            ik_glob = local_indices[ik_loc]
            h_mat = h_all_local[ik_loc]
            w, v = np.linalg.eigh(h_mat)

            evals_local[:, ik_loc] = w

            # Unfolding projection onto reference primitive collinear model
            e_1k = w[:, None]
            v_1k = v[:, None, :]
            k_s_1k = k_sc_input[ik_glob:ik_glob + 1]

            proj_1k = unfold_spin_project.project_sc_bands_on_reference(
                tb_up=tb_up,
                tb_dn=tb_dn,
                e_sc=e_1k,
                v_sc=v_1k,
                k_s_list=k_s_1k,
                sc_red_lat=sc_red_lat,
                u_samples=u_samples,
                fermi_level=args.fermi_level,
                nb_ref_keep=2 * tb_up._norb
            )

            # Match primitive k-point sector
            k_p_prim = proj_1k["k_p"][0] / np.diag(sc_red_lat)
            diffs = np.linalg.norm(k_p_prim - k_vec[ik_glob], axis=-1)
            iq_match = np.argmin(diffs)

            w_pick_local[:, ik_loc] = proj_1k["weights"][:, 0, iq_match, :].sum(axis=-1)
            s_lab_local[:, ik_loc, :] = proj_1k["spin_lab"][:, 0, iq_match, :, :].sum(axis=-2)
            s_loc_local[:, ik_loc, :] = proj_1k["spin_local"][:, 0, iq_match, :, :].sum(axis=-2)

            del w, v, e_1k, v_1k, proj_1k

    if comm is not None:
        comm.Barrier()
    t_solve = time.perf_counter() - t0_solve

    # 10. Gather results onto rank 0
    if comm is not None and size > 1:
        gathered_evals = comm.gather(evals_local, root=0)
        gathered_w = comm.gather(w_pick_local, root=0)
        gathered_s_lab = comm.gather(s_lab_local, root=0)
        gathered_s_loc = comm.gather(s_loc_local, root=0)
    else:
        gathered_evals = [evals_local]
        gathered_w = [w_pick_local]
        gathered_s_lab = [s_lab_local]
        gathered_s_loc = [s_loc_local]

    if rank == 0:
        evals = np.concatenate(gathered_evals, axis=1)    # (nband_sc, nk)
        w_pick = np.concatenate(gathered_w, axis=1)       # (nband_sc, nk)
        s_lab = np.concatenate(gathered_s_lab, axis=1)    # (nband_sc, nk, 3)
        s_loc = np.concatenate(gathered_s_loc, axis=1)    # (nband_sc, nk, 3)
        s_pick = s_loc if args.spin_frame == "local" else s_lab

        print(f"Parallel solve & unfolding completed in {t_solve:.2f} s across {size} MPI rank(s).")
        print(f"  Eigenvalues shape: {evals.shape}")
        print(f"  Weights shape    : {w_pick.shape}")
        print(f"  Spin proj shape  : {s_pick.shape} (frame: {args.spin_frame})")

        t0_post = time.perf_counter()

        # 11. Compute Total Density of States (DOS)
        print("Computing Total Density of States (DOS)...")
        energy_dos_raw, total_dos = compute_total_dos_from_evals(
            evals=evals,
            sigma=0.02,
            n_energy=2000
        )
        energy_dos = energy_dos_raw - args.fermi_level

        # 12. Spectral broadening and RGB spin mapping
        print("Broadening spectral function and evaluating spin RGB mapping...")
        e_grid, a, axyz = gaussian_spectral_from_bands(
            E=evals - args.fermi_level,
            w=w_pick,
            s=s_pick,
            sigma=0.08,
            nE=1000
        )

        eps = 1e-12
        rgb = np.abs(axyz)
        rgb /= np.maximum(np.sum(rgb, axis=2, keepdims=True), eps)
        intensity = a / (a.max() + eps)
        img = np.clip(rgb * intensity[..., None], 0.0, 1.0)

        # 13. Save complete numerical output for offline plotting (.npz)
        out_file = args.output_data
        if not out_file.endswith(".npz"):
            out_file += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        print(f"Saving complete band and unfolding data to {out_file}...")
        np.savez_compressed(
            out_file,
            evals=evals,
            weights=w_pick,
            spin_proj=s_pick,
            spin_lab=s_lab,
            spin_local=s_loc,
            k_vec=k_vec,
            x_vec=x_vec,
            labels=labels,
            tick_pos=tick_pos,
            fermi_level=args.fermi_level,
            sigma=0.08,
            e_grid=e_grid,
            spectral_weight=a,
            spin_xyz=axyz,
            img_rgb=img,
            energy_dos=energy_dos,
            total_dos=total_dos,
            supercell=np.array([args.nx, args.ny, args.nz]),
            texture=args.texture,
            r0=args.r0,
            lx=args.lx,
            spin_frame=args.spin_frame,
        )
        print(f"  [SAVED] {out_file}")

        # 14. Save PNG figures if requested
        if args.save_png:
            os.makedirs(args.png_dir, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Plot 1: Unfolded Spin-Projected Bands
            unfold_png_path = os.path.join(args.png_dir, "unfolded_spin_projected_bands.png")
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
            ax.imshow(
                np.transpose(img, (1, 0, 2)),
                origin="lower",
                aspect="auto",
                extent=[x_vec.min(), x_vec.max(), e_grid.min(), e_grid.max()]
            )
            for xp in tick_pos:
                ax.axvline(xp, color="w", lw=0.6, alpha=0.5)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel(r"$E - E_F$ (eV)")
            frame_label = "Local-Frame" if args.spin_frame == "local" else "Lab-Frame"
            ax.set_title(f"{frame_label} Spin-Projected Unfolded Bands (RGB: $|S_x|, |S_y|, |S_z|$)")
            fig.tight_layout()
            fig.savefig(unfold_png_path, dpi=300)
            plt.close(fig)
            print(f"  [SAVED] {unfold_png_path}")

            # Plot 2: Supercell Band Structure
            bands_png_path = os.path.join(args.png_dir, "supercell_bands.png")
            fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
            for n in range(evals.shape[0]):
                ax.plot(x_vec, evals[n, :] - args.fermi_level, color="black", lw=0.8, alpha=0.85)
            for xp in tick_pos:
                ax.axvline(xp, color="gray", lw=0.7, alpha=0.6, linestyle="--")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel(r"$E - E_F$ (eV)")
            ax.set_xlabel(r"$k$-path")
            ax.set_title(f"Supercell Bands ({args.nx}x{args.ny}x{args.nz})")
            fig.tight_layout()
            fig.savefig(bands_png_path, dpi=300)
            plt.close(fig)
            print(f"  [SAVED] {bands_png_path}")

            # Plot 3: DOS
            dos_png_path = os.path.join(args.png_dir, "total_dos.png")
            fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
            ax.plot(energy_dos, total_dos, color="black", lw=1.8, label="Total DOS")
            ax.axvline(0.0, color="red", linestyle=":", lw=1.0, label=r"$E_F$")
            ax.set_xlabel(r"Energy $E - E_F$ (eV)")
            ax.set_ylabel("DOS (states / eV)")
            ax.set_title("Electronic Density of States")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(dos_png_path, dpi=300)
            plt.close(fig)
            print(f"  [SAVED] {dos_png_path}")

        t_post = time.perf_counter() - t0_post
        t_wall_total = time.perf_counter() - t_wall_start

        # 15. Save detailed timing benchmark information
        timing_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_mpi_procs": size,
            "supercell": [args.nx, args.ny, args.nz],
            "norb_unitcell": 2 * tb_up._norb,
            "norb_supercell": nband_sc,
            "nk_points": nk_total,
            "texture": args.texture,
            "lx": args.lx,
            "r0": args.r0,
            "time_setup_seconds": round(t_setup, 3),
            "time_solver_unfolding_seconds": round(t_solve, 3),
            "time_postprocessing_plot_seconds": round(t_post, 3),
            "time_total_seconds": round(t_wall_total, 3),
            "time_per_kpoint_seconds": round(t_solve / nk_total, 4)
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.timing_file)), exist_ok=True) if os.path.dirname(args.timing_file) else None
        with open(args.timing_file, "w") as f:
            json.dump(timing_info, f, indent=2)
        print(f"  [SAVED] {args.timing_file}")

        txt_file = os.path.splitext(args.timing_file)[0] + ".txt"
        with open(txt_file, "w") as f:
            f.write("=" * 65 + "\n")
            f.write("  BENCHMARK & TIMING SUMMARY - magn_texture_tb_mpi.py\n")
            f.write("=" * 65 + "\n")
            f.write(f"Timestamp                 : {timing_info['timestamp']}\n")
            f.write(f"MPI Communicator Size     : {timing_info['num_mpi_procs']} rank(s)\n")
            f.write(f"Supercell Dimensions      : {timing_info['supercell']}\n")
            f.write(f"Supercell Hamiltonian Dim : {timing_info['norb_supercell']} x {timing_info['norb_supercell']}\n")
            f.write(f"Total k-points            : {timing_info['nk_points']}\n")
            f.write(f"Magnetic Texture          : {timing_info['texture']} (lx={timing_info['lx']}, r0={timing_info['r0']})\n")
            f.write("-" * 65 + "\n")
            f.write(f"Setup & Supercell Time    : {timing_info['time_setup_seconds']:.2f} s\n")
            f.write(f"Parallel Solve & Unfold   : {timing_info['time_solver_unfolding_seconds']:.2f} s\n")
            f.write(f"Postprocessing & Plotting : {timing_info['time_postprocessing_plot_seconds']:.2f} s\n")
            f.write(f"Total Wall Clock Time     : {timing_info['time_total_seconds']:.2f} s\n")
            f.write(f"Average Time per k-point  : {timing_info['time_per_kpoint_seconds']:.4f} s\n")
            f.write("=" * 65 + "\n")
        print(f"  [SAVED] {txt_file}")

        print("=" * 75)
        print("Calculations complete successfully.")
        print("=" * 75)


if __name__ == "__main__":
    main()
