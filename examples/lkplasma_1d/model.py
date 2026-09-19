import os
import os

if __name__=="__main__":
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="PetraM sciprt")
    parser.add_argument("-s", "--force-serial", 
                     action = "store_true", 
                     default = True,
                     help="Use serial model even if nproc > 1.")
    parser.add_argument("-p", "--force-parallel", 
                     action = "store_true", 
                     default = False,
                     help="Use parallel model even if nproc = 1.")
    parser.add_argument("-d", "--debug-param", 
                     action = "store", 
                     default = 1, type=int) 

    args = parser.parse_args()
    if args.force_parallel:
        use_parallel = True
        args.force_serial = False
    else:
        use_parallel = False

    import  petram.mfem_config as mfem_config
    mfem_config.use_parallel = use_parallel
    debug_level=args.debug_param

    os.environ["PETRAM_ARRAY_ID"] = "0"
    os.environ["PETRAM_ARRAY_COUNT"] = "1"

# this is needed if this file is being imported
if not "use_parallel" in locals():
    use_parallel = False
#set default parallel/serial flag
if use_parallel:
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
else:
    myid = 0
    num_proc = 1

from numpy import sin
from numpy import cos
from numpy import tan
from petram.helper.variables import cosd
from petram.helper.variables import sind
from petram.helper.variables import tand
from numpy import arctan
from numpy import arctan2
from numpy import exp
from numpy import log10
from numpy import log
from numpy import log2
from numpy import sqrt
from numpy import abs
from numpy import conj
from numpy import real
from numpy import imag
from numpy import sum
from numpy import dot
from numpy import vdot
from numpy import array
from numpy import cross
from numpy import min
from numpy import sign
from numpy import ones
from numpy import eye
from numpy import diag
from numpy import zeros
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import linspace
from numpy import logspace
from numpy import hstack
from numpy import vstack
from numpy import dstack
from numpy import stack
from petram.solver.parametric_scanner import Scan

from petram.mfem_model import MFEM_ModelRoot
from petram.mfem_model import MFEM_GeneralRoot
from petram.mfem_model import MFEM_GeomRoot
from petram.mfem_model import MFEM_MeshRoot
from petram.mesh.mesh_model import MFEMMesh
from petram.mesh.mesh_model import Mesh1D
from petram.mesh.mesh_model import UniformRefinement
from petram.mfem_model import MFEM_PhysRoot
from petram.phys.em1d.em1d_model import EM1D
from petram.phys.em1d.em1d_model import EM1D_DefDomain
from petram.phys.em1d.em1d_vac import EM1D_Vac
from petram.phys.em1d.em1d_coldplasma import EM1D_ColdPlasma
from petram.phys.em1d.em1d_lkplasma import EM1D_LocalKPlasma
from petram.phys.em1d.em1d_model import EM1D_DefBdry
from petram.phys.em1d.em1d_pec import EM1D_PEC
from petram.phys.em1d.em1d_cont import EM1D_Continuity
from petram.phys.em1d.em1d_e import EM1D_E
from petram.phys.em1d.em1d_port import EM1D_Port
from petram.phys.em1d.em1d_model import EM1D_DefPair
from petram.mfem_model import MFEM_InitRoot
from petram.init_model import InitSetting
from petram.mfem_model import MFEM_PostProcessRoot
from petram.postprocess.dxp_model import DataExport
from petram.postprocess.pc_export import PointCloud
from petram.mfem_model import MFEM_SolverRoot
from petram.solver.solver_model import SolveStep
from petram.solver.std_solver_model import StdSolver
from petram.solver.superlu_model import SuperLU
from petram.solver.mumps_model import MUMPS
from collections import OrderedDict

def make_model():
    obj1 = MFEM_ModelRoot()
    obj1.root_path = '/tmp/piscope_shiraiwa/bastet01.pid3689161/.###ifigure_home_shiraiwa_src_TwoPiTest_piscope_projects_em1d_lkplasma.pfz/proj/model1/mfem'
    obj1.model_path = '/home/shiraiwa/src/TwoPiTest/piscope_projects'
    obj1.pkg_versions = {'petram_rf': '26.2.7', 'petram': '26.8.29', 'petram_extraphysnumba': '25.11.28', 'petram_rfp': '26.7.7', 'petram_extraphys': '25.11.28', 'petram_dpg': '26.6.29', 'petram_ds': '1.2.0', 'petram_whp': '25.11.28'}
    obj2 = obj1.add_node(name = "General", cls = MFEM_GeneralRoot)
    obj2.allow_fallback_nonjit = 'warn' 
    obj2.savegz = 'off'
    obj2.ns_name = "global"
    obj3 = obj1.add_node(name = "Geometry", cls = MFEM_GeomRoot)
    obj4 = obj1.add_node(name = "Mesh", cls = MFEM_MeshRoot)
    obj5 = obj4.add_node(name = "MeshGroup1", cls = MFEMMesh)
    obj6 = obj5.add_node(name = "Mesh1D1", cls = Mesh1D)
    obj6.enabled = False
    obj6.length_txt = '0.1, 0.42, 0.1'
    obj6.nsegs_txt = '80,  336,  80'
    obj6.mesh_x0_txt = '0.3762'
    obj7 = obj5.add_node(name = "Mesh1D2", cls = Mesh1D)
    obj7.length_txt = '0.1, 0.42, 0.1'
    obj7.nsegs_txt = '40,  160,  40'
    obj7.mesh_x0_txt = '0.37'
    obj8 = obj5.add_node(name = "Mesh1D3", cls = Mesh1D)
    obj8.enabled = False
    obj8.length_txt = '0.1, 0.42, 0.1'
    obj8.nsegs_txt = '120,  504,  120'
    obj8.mesh_x0_txt = '0.3762'
    obj9 = obj5.add_node(name = "UniformRefinement1", cls = UniformRefinement)
    obj9.num_refine = '3'
    obj10 = obj1.add_node(name = "Phys", cls = MFEM_PhysRoot)
    obj11 = obj10.add_node(name = "EM1D1", cls = EM1D)
    obj11.dep_vars_suffix = '1'
    obj11.order_txt = 'order'
    obj11.freq_txt = 'freq'
    obj12 = obj11.add_node(name = "Domain", cls = EM1D_DefDomain)
    obj12.ky_txt = 0
    obj12.kz_txt = 0
    obj12.Einit_m_txt = '[0, 0, 0]'
    obj13 = obj12.add_node(name = "Vac2", cls = EM1D_Vac)
    obj13.sel_index_txt = '3, 1'
    obj13.epsilonr_txt = '(1+0j)'
    obj13.mur_txt = '(1+0j)'
    obj13.sigma_txt = '0j'
    obj13.ky_txt = 'ky'
    obj13.kz_txt = 'kz'
    obj13.Einit_m_txt = '[0, 0, 0]'
    obj14 = obj12.add_node(name = "ColdPlasma1", cls = EM1D_ColdPlasma)
    obj14.enabled = False
    obj14.sel_index_txt = '2'
    obj14.bext_txt = '=(0,0, bnorm_jit)'
    obj14.dens_e_txt = '=dens_jit'
    obj14.temperature_txt = '30'
    obj14.dens_i_txt = '=dens_jit*fraci, dens_jit*fracim'
    obj14.mass_txt = 'Ai, Aim'
    obj14.charge_q_txt = 'Zi, Zim'
    obj14.ky_txt = 'ky'
    obj14.kz_txt = 'kz'
    obj14.Einit_m_txt = '[0, 0, 0]'
    obj14.stix_terms = '(default) include all'
    obj14.col_model = 'w/o col.'
    obj15 = obj12.add_node(name = "LocalKPlasma1", cls = EM1D_LocalKPlasma)
    obj15.sel_index_txt = '2'
    obj15.bext_txt = '=(0,0, bnorm_jit)'
    obj15.dens_e_txt = '=dens_jit'
    obj15.temperature_e_txt = '=te_jit'
    obj15.dens_i_txt = '=dens_jit*fraci, dens_jit*fracim'
    obj15.temperatures_i_txt = '=ti_jit,  ti_jit*5'
    obj15.temperatures_c_txt = '500.0'
    obj15.mass_txt = '2, 1'
    obj15.charge_q_txt = '1, 1'
    obj15.kpa_kpe_txt = '=kz, 1.'
    obj15.kpe_vec_txt = '1, 0, 0'
    obj15.ky_txt = 'ky'
    obj15.kz_txt = 'kz'
    obj15.Einit_m_txt = '[0, 0, 0]'
    obj15.kpe_mode = 'fast wave'
    obj15.kpe_alg = 'em1d'
    obj15.lk_terms = '0\nelectrons,Sig:1, Del:1, Pi:1, Tau:1, Eta:1, Xi:1\nions1,Sig:1, Del:1, Pi:1, Tau:1, Eta:1, Xi:1\nions2,Sig:1, Del:1, Pi:1, Tau:1, Eta:1, Xi:1'
    obj16 = obj11.add_node(name = "Boundary", cls = EM1D_DefBdry)
    obj16.Einit_m_txt = '[0, 0, 0]'
    obj17 = obj16.add_node(name = "PEC1", cls = EM1D_PEC)
    obj17.sel_index_txt = '1'
    obj17.Einit_m_txt = '[0, 0, 0]'
    obj18 = obj16.add_node(name = "Continuity1", cls = EM1D_Continuity)
    obj18.sel_readonly = False
    obj18.sel_index_txt = '2, 3'
    obj19 = obj16.add_node(name = "E1", cls = EM1D_E)
    obj19.enabled = False
    obj19.sel_index_txt = '1'
    obj19.E_z_txt = '1'
    obj19.E_m_txt = '[0, 0]'
    obj19.Einit_m_txt = '[0, 0, 0]'
    obj20 = obj16.add_node(name = "Port1", cls = EM1D_Port)
    obj20.sel_index_txt = '4'
    obj20.inc_amp_y_txt = '1'
    obj20.inc_amp_z_txt = '0'
    obj20.inc_amp_m_txt = '[1., 0.]'
    obj20.inc_phase_txt = '0.0'
    obj20.epsilonr_txt = '(1+0j)'
    obj20.mur_txt = '(1+0j)'
    obj20.ky_txt = 'ky'
    obj20.kz_txt = 'kz'
    obj20.Einit_m_txt = '[0, 0, 0]'
    obj20.port_idx = '1'
    obj21 = obj16.add_node(name = "Port2", cls = EM1D_Port)
    obj21.enabled = False
    obj21.sel_index_txt = '1'
    obj21.inc_amp_y_txt = '0.0'
    obj21.inc_amp_z_txt = '1.0'
    obj21.inc_amp_m_txt = '[1., 0.]'
    obj21.inc_phase_txt = '0.0'
    obj21.epsilonr_txt = '(1+0j)'
    obj21.mur_txt = '(1+0j)'
    obj21.ky_txt = 'ky'
    obj21.kz_txt = 'kz'
    obj21.Einit_m_txt = '[0, 0, 0]'
    obj21.isTimeDependent_RHS = False
    obj21.port_idx = '2'
    obj22 = obj11.add_node(name = "Pair", cls = EM1D_DefPair)
    obj23 = obj1.add_node(name = "InitialValue", cls = MFEM_InitRoot)
    obj24 = obj23.add_node(name = "InitSetting1", cls = InitSetting)
    obj24.phys_model = 'EM1D1'
    obj24.init_mode = 1
    obj24.init_value_txt = '1'
    obj25 = obj1.add_node(name = "PostProcess", cls = MFEM_PostProcessRoot)
    obj26 = obj25.add_node(name = "DataExport1", cls = DataExport)
    obj26.use_scanner = False
    obj27 = obj26.add_node(name = "PointCloud1", cls = PointCloud)
    obj27.pc_x_txt = 'linspace(0.37, 0.99, 200)'
    obj27.pc_y_txt = 'zeros(200)'
    obj27.pc_z_txt = 'zeros(200)'
    obj27.export_expr = 'Pabsi21'
    obj28 = obj1.add_node(name = "Solver", cls = MFEM_SolverRoot)
    obj29 = obj28.add_node(name = "SolveStep1", cls = SolveStep)
    obj29.postprocess_sol = 'DataExport1'
    obj30 = obj29.add_node(name = "StdSolver1", cls = StdSolver)
    obj30.phys_model = 'EM1D1'
    obj31 = obj30.add_node(name = "SuperLU1", cls = SuperLU)
    obj32 = obj28.add_node(name = "SolveStep2", cls = SolveStep)
    obj32.init_setting = 'InitSetting1'
    obj32.enabled = False
    obj33 = obj32.add_node(name = "StdSolver2", cls = StdSolver)
    obj33.phys_model = 'EM1D1'
    obj34 = obj33.add_node(name = "MUMPS1", cls = MUMPS)
    return obj1

if __name__ == "__main__":
    if (myid == 0): parser.print_options(args)

    import time, datetime
    stime = time.time()
    
    if mfem_config.use_parallel:
        from petram.engine import ParallelEngine as Eng
    else:
        from petram.engine import SerialEngine as Eng
    
    import petram.debug as debug
    debug.set_debug_level(debug_level)
    
    model = make_model()
    
    eng = Eng(model = model)
    eng.show_environment()
    
    solvers = eng.run_build_ns()
    
    is_first = True
    for s in solvers:
        s.run(eng, is_first=is_first)
        is_first=False
    
    eng.show_variables()
    if myid == 0:
        print("End Time " + 
              datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
        print("Total Elapsed Time: " + str(time.time()-stime) + "s")
        print("Petra-M Normal End")