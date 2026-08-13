###
#Timestream Simulation Script for Prime-cam
###
###Last updated: Feb 04, 2025
###
#Author: Ankur Dev, adev@astro.uni-bonn-de
###
###Logbook###
###
###Personal Notes and Updates:
#ToDo: Implement config and CLI
#ToDo: Implement CAR
#ToDo: Verify and validate sequential steps
###
###[Self]Updates Log:
#20-02-2024: Scraping TOAST2, Begin migration
#14-03-2024: Upgraded to TOAST3
#19-03-2024: Checked write and read to h5
#21-03-2024: Implement dual-regime Atm sim
#22-03-2024: Begin integrated script implemention
#23-03-2024: Implemented multiple schedules
#23-03-2024: Modified schedule field name; must be %field_%dd_%mm for uid
#02-04-2024: Updated Atmosphere implementation
#25-06-2024: Implementing tests for sinosoidal modulation
#14-09-2024: Implementing changes for mpirun
#15-09-2024: primecam_mockdata_pipeline func takes now schedule object rather than sch file path
#15-09-2024: Implemented group size calculation and accept user values for MPI
#16-09-2024: Script now accepts schedule file name as argument
#15-10-2024: Updated class Args, implemented number of detectors as arg requirement
#15-10-2024: Updated h5_outdir to include ndets info in path
#22-01-2025: Updated focalplane file to h5 format, TOAST3 format
#04-02-2025: Updated gains for atm sim
#21-09-2025: Updated all dets to correct for NET values
#21-09-2025: Updated gains for atm sim
#13-08-2026: Updated fp_trim and mockdata_pipeline to include full module selection
###

"""
Description:
Timestream Simulation Script for Prime-cam.
This script is modified from TOAST Ground Simulation to perform timestream simulation
for Prime-Cam with FYST. Note: this is a work in progress.

The script demonstrates an example workflow, from generating loading detectors, scanning
an input map, making detailed atmospheric simulation and generating mock detector timestreams. 
for more complex simulations tailored to specific experimental needs.

Usage:
Ref: https://github.com/hpc4cmb/toast/blob/toast3/workflows/toast_sim_ground.py
Ref: TOAST3 Documentation: https://toast-cmb.readthedocs.io/en/toast3/intro.html

"""

import toast
import toast.io as io
import toast.ops
from toast.mpi import MPI
from scripts.helper_scripts.calc_groupsize import job_group_size, estimate_group_size
from scripts.fp_scripts import fp_trim
from scripts.sso_scripts.sim_sso import SimSSO

import astropy.units as u
from astropy.table import QTable, Column


import numpy as np
from datetime import datetime
import os
import argparse
import random
import time as t

# Define the global args class
class Args:
    def __init__(self, parsed_args):   
        # Keep input sky map False for planet sims; planet done by SSO
        self.scan_inmap = False #False Default # Toggle to Scan Input Sky Map
        self.sim_atm = True #True Default # Toggle to Simulate Atmosphere
        self.sim_noise = True #True Default # Toggle to Simulate Detector Noise

        self.weather = 'atacama'
        self.sample_rate = 488 * u.Hz #488 Hz # or 244 Hz
        self.scan_rate_az = 0.2  * (u.deg / u.s) #on sky rate , or 1 deg/s
        #fix_rate_on_sky (bool):  If True, `scan_rate_az` is given in sky coordinates and azimuthal
        #rate on mount will be adjusted to meet it.
        #If False, `scan_rate_az` is used as the mount azimuthal rate. (default = True)

        self.scan_accel_az = 1  * (u.deg / u.s**2) # or 4 deg/s^2
        self.fov = 1.3 * u.deg # Field-of-view in degrees
        # g3_outdir = "./g3_dataframes"
        self.h5_outdir = os.path.join(
            ".", "ccat_datacenter_mock", 
            "mockdata", 
            f"planet_ATMdata_d{parsed_args.dets}"
        )
        
        self.mode = "IQU" #"IQU"
        self.input_map = "pysm3_map_nside2048_allStokes.fits"
        self.nside = 2048 #1024
        self.freq = 280 * u.GHz
        self.fwhm = 0.78 *u.arcmin


def primecam_mockdata_pipeline(args, comm, focalplane, schedule, group_size):
    #Set up logger and timer
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()
    
    if group_size is not None:
        # Create the toast communicator with specified group size
        toast_comm = toast.Comm(world=comm, groupsize=group_size)
        log.info_rank(f"Job group size: {toast_comm.group_size}", comm)
        log.info_rank(f"Number of process groups: {toast_comm.ngroups}", comm)
    else:
        log.info_rank(f"Begin job planning ...", comm)
        # grp_size_calc = job_group_size(world_comm=comm, schedule=schedule,
        #                                focalplane=focalplane)
        grp_size_calc = estimate_group_size(world_comm=comm, schedule=schedule)
        # Create the toast communicator
        toast_comm = toast.Comm(world=comm, groupsize=grp_size_calc)
        log.info_rank(f"Job group size: {toast_comm.group_size}", comm)
        log.info_rank(f"Number of process groups: {toast_comm.ngroups}", comm)
    
    # Shortcut for the world communicator
    world_comm = toast_comm.comm_world

    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
        weather=args.weather,
    )
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    log.info_rank(f"Telescope metadata: \n {telescope}", world_comm)
    log.info_rank(f"FocalPlane: {focalplane}", world_comm)

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    #Load SimGround

    sim_ground = toast.ops.SimGround(weather=args.weather)
    sim_ground.telescope = telescope
    sim_ground.schedule = schedule
    sim_ground.scan_rate_az =  args.scan_rate_az
    sim_ground.scan_accel_az = args.scan_accel_az
    sim_ground.max_pwv = 1.41 *u.mm
    
    #=============================#
    ### El Nod Tests ###

    # sim_ground.scan_rate_el = 1.5 * (u.deg / u.s) #rate allowed by mount
    # sim_ground.el_mod_amplitude = 0.2 * u.deg #1.0
    # sim_ground.el_mod_rate = 1 * u.Hz #10 sec sinosoid, 0.1
    # sim_ground.el_mod_sine = True

    # sim_ground.elnod_every_scan = False
    #=============================#

    sim_ground.apply(data)
    
    log.info_rank(f"Number of Observations loaded: {len(data.obs)}", world_comm)

    #Noise Model
    # Construct a "perfect" noise model just from the focalplane parameters
    default_model = toast.ops.DefaultNoiseModel()
    default_model.apply(data)

    #Detector Pointing
    det_pointing_azel = toast.ops.PointingDetectorSimple(quats="quats_azel")
    det_pointing_azel.boresight = sim_ground.boresight_azel

    det_pointing_radec = toast.ops.PointingDetectorSimple(quats="quats_radec")
    det_pointing_radec.boresight = sim_ground.boresight_radec

    # Create the Elevation modulated noise model
    elevation_noise = toast.ops.ElevationNoise(out_model="el_noise_model")
    elevation_noise.detector_pointing = det_pointing_azel
    elevation_noise.apply(data)

    ### Pointing Matrix
    # Set up the pointing matrix in RA / DEC, and pointing weights in Az / El
    # in case we need that for the atmosphere sim below.
    pixels_radec = toast.ops.PixelsHealpix(
        detector_pointing=det_pointing_radec,
        nside=args.nside,
    )

    weights_radec = toast.ops.StokesWeights(weights="weights_radec", mode=args.mode)
    weights_radec.detector_pointing = det_pointing_radec
    weights_azel = toast.ops.StokesWeights(weights="weights_azel", mode=args.mode)
    weights_azel.detector_pointing = det_pointing_azel
    log.info_rank(" Simulated telescope boresight pointing in", comm=world_comm, timer=timer)

    ### Input Map Signal
    # hp_input_map = os.path.join("input_files", "pysm3_map_nside2048.fits")
    # Full Stokes
    
    if args.scan_inmap:
        hp_input_map = os.path.join("input_files", args.input_map)
        #check if this file exists, else raise runtime error
        if not os.path.exists(hp_input_map):
            raise RuntimeError(f"Input map file not found: {hp_input_map}")
        
        scan_map = toast.ops.ScanHealpixMap(file=hp_input_map)
        scan_map.enabled = args.scan_inmap  
        scan_map.pixel_pointing = pixels_radec
        scan_map.stokes_weights = weights_radec
        scan_map.apply(data)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After Scanning Input Map:  {mem}", world_comm)
    
    #-------------------------------------------------------#
    log.info_rank(f"Begin SSO Observation...", world_comm)
    sso = SimSSO(
                sso_name="Jupiter",
                beam_file="scripts/sso_scripts/data/gauss_beam.h5",
                detector_pointing=det_pointing_azel,   # az-el detector pointing
                finite_sso_radius=True, # set True if source not point-like
            )
    sso.apply(data)
    log.info_rank(" Simulated SSO signal in", comm=world_comm, timer=timer)
    #--------------------------------------------------------#

    ### Atmospheric simulation
    log.info_rank(f"Atmospheric simulation...", world_comm)
    #Atmosphere set-up
    rand_realisation = random.randint(10000, 99999)
    tel_fov = 1.5* u.deg # 4* u.deg , changed 17.02.2025
    cache_dir = None
    # cache_dir = "./atm_cache"

    sim_atm_coarse =toast.ops.SimAtmosphere(
                    name="sim_atm_coarse",
                    add_loading=False,
                    lmin_center=300 * u.m,
                    lmin_sigma=30 * u.m,
                    lmax_center=10000 * u.m,
                    lmax_sigma=1000 * u.m,
                    xstep=50 * u.m,
                    ystep=50 * u.m,
                    zstep=50 * u.m,
                    zmax=2000 * u.m,
                    nelem_sim_max=30000,
                    gain=1e-4, #1e-5, changed 20.09.2025
                    realization=1000000,
                    wind_dist=10000 * u.m,
                    enabled=False,
                    cache_dir=cache_dir,
                )

    sim_atm_coarse.realization = 1000000 + rand_realisation
    sim_atm_coarse.field_of_view = tel_fov
    sim_atm_coarse.detector_pointing = det_pointing_azel
    sim_atm_coarse.enabled = args.sim_atm  # Toggle to False to disable
    sim_atm_coarse.serial = False
    sim_atm_coarse.apply(data)
    log.info_rank(" Applied large-scale Atmosphere simulation in", comm=world_comm, timer=timer)

    sim_atm_fine= toast.ops.SimAtmosphere(
            name="sim_atm_fine",
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=4 * u.m,
            ystep=4 * u.m,
            zstep=4 * u.m,
            zmax=200 * u.m, #changed 31.01.2025
            gain=4e-5, #1e-5, changed 20.09.2035
            wind_dist=1000 * u.m,
            enabled=False,
            cache_dir=cache_dir,
        )

    sim_atm_fine.realization = rand_realisation
    sim_atm_fine.field_of_view = tel_fov
    
    sim_atm_fine.detector_pointing = det_pointing_azel
    sim_atm_fine.enabled = args.sim_atm  # Toggle to False to disable
    sim_atm_fine.serial = False
    sim_atm_fine.apply(data)

    log.info_rank("Applied small-scale Atmosphere simulation in", comm=world_comm, timer=timer)
    #------------------------#
    
    #simulate detector noise
    sim_noise = toast.ops.SimNoise()
    sim_noise.noise_model = elevation_noise.out_model
    sim_noise.enabled = args.sim_noise
    sim_noise.serial = False
    sim_noise.apply(data)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After generating detector timestreams:  {mem}", world_comm)

    field_name = (data.obs[0].name).split('-')[0]
    n_dets = telescope.focalplane.n_detectors
    #start_unix = data.obs[0].shared["times"][0]
    #end_unix = data.obs[len(data.obs)-1].shared["times"][0]
    #format_obs_startend = format_unix_times(start_unix, end_unix)

    ### Write to h5
    output_dir = args.h5_outdir
    module_code = args.freq.to_string().split('.0')[0]
    f_path = f"sim_PCAM{module_code}_h5_{field_name}_d{n_dets}"
    save_dir = os.path.join(args.h5_outdir, f_path)
    os.makedirs(save_dir, exist_ok=True)

    log.info_rank(f"Writing timestream data to h5 files for \
Field {field_name} observed at {args.freq.to_string()} \
with {n_dets} detectors", world_comm)
    log.info_rank(f"Writing h5 files to: {save_dir}", world_comm)

    detdata_tosave = ["signal", "flags"]

    for obs in data.obs:
        io.save_hdf5(
            obs=obs,
            dir=save_dir,
            detdata=detdata_tosave
        )

###==================================================###    

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Simulate PrimeCam Timestream Data with 1 schedule",
        epilog="Note: Provide only the name of the schedule file, not the path. \n \
            The schedule file must be in the 'input_files/schedules' directory.")
    # Required argument for the schedule file
    parser.add_argument('-s','--sch', required=True, help="Name of the schedule file")
    parser.add_argument('-d', '--dets', 
                        default="100", 
                        help="Number of detectors, or FULL for no trimming")
    parser.add_argument('--module-arrays', '--MODULE_ARRAYS',
                        dest='MODULE_ARRAYS',
                        type=int,
                        default=int(os.environ.get("MODULE_ARRAYS", 1)),
                        choices=[1, 3],
                        help="Module arrays to use: 1 selects w2, 3 selects w1/w2/w3 detectors")
    parser.add_argument('-g','--grp_size', default=None, type=int, help="Group size (optional)")

    parsed_args = parser.parse_args()
    
    args = Args(parsed_args)
    
    #Set up logger and timer
    log_global = toast.utils.Logger.get()
    global_timer = toast.timing.Timer()
    timer = toast.timing.Timer()
    global_timer.start()
    timer.start()

    # Initialize the communicator
    comm, procs, rank = toast.get_world()

    # Focalplane file preparation 
    # Rank 0 prepares the focalplane file once; other ranks wait and reuse it.
    if rank == 0:
        ndets_selected, fp_filename = fp_trim.build_fp_file(
            parsed_args.dets,
            module_arrays=parsed_args.MODULE_ARRAYS,
        )
    else:
        ndets_selected, fp_filename = None, None

    if comm is not None:
        ndets_selected = comm.bcast(ndets_selected, root=0)
        fp_filename = comm.bcast(fp_filename, root=0)
        comm.barrier()

    # Keep detector count consistent with the exact focalplane file used.
    parsed_args.dets = ndets_selected
    args = Args(parsed_args)
    
    # Initialize the TOAST logger
    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = "unknown number of"
        
    log_global.info_rank(
        f"Executing PrimeCam workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads at {datetime.now()}",
        comm,
    )

    log_global.info_rank(
        f"Using TOAST version: {toast.__version__}", comm)

    log_global.info_rank(
        f"Using {parsed_args.MODULE_ARRAYS} module array(s) with {parsed_args.dets} detectors",
        comm,
    )

    log_global.info_rank(
        f"Starting timesteam simulation...", comm)
    if rank == 0:
        sim_start_time = t.time()

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log_global.info_rank(f"Start of the workflow:  {mem}", comm)
    
    log_global.info_rank(
        f"Begin set-up and monitors for Simulating timestream data for PrimeCam/FYST",
        comm)

    # Load the rank-synchronized focalplane file.
    try:
        det_table = QTable.read(fp_filename, path='dettable_trim')
    except Exception as e:
        log_global.error(f"Failed to load focalplane file: {fp_filename}. Error: {e}", comm)
        raise  

    
    log_global.info_rank(f"Loading focalplane: {fp_filename}", comm)
    
    # instantiate a TOAST focalplane instance 
    width = args.fov
    focalplane = toast.instrument.Focalplane(
        detector_data=det_table,
        sample_rate=args.sample_rate,
        field_of_view=1.1 * (width + 2 * args.fwhm),
    )

    # Load the schedule file and instantiate the schedule object
    schedule_file = os.path.join("input_files/schedules",parsed_args.sch)
    schedule = toast.schedule.GroundSchedule()
    schedule.read(schedule_file, comm=comm)
    
    # Run the simulation pipeline
    primecam_mockdata_pipeline(args, comm, focalplane, schedule, parsed_args.grp_size)
    log_global.info_rank(f"Wrote timestream data for {schedule_file} to disk", comm=comm)

    
    log_global.info_rank("Full mock data generated in", comm=comm, timer=global_timer)
    
    # Synchronize all ranks, so every process reaches this point before proceeding
    comm.barrier()
    if rank == 0:
        sim_end_time = t.time()
        sim_elapsed_time = sim_end_time - sim_start_time
        log_global.info_rank(
            f"Timestream Simulation completed. Elapsed Time: {sim_elapsed_time/60.0:.2f} minutes",
            comm
            )
    
if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
