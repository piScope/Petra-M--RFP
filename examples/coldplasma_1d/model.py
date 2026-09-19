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
from petram.mfem_model import MFEM_PhysRoot
from petram.phys.em1d.em1d_model import EM1D
from petram.phys.em1d.em1d_model import EM1D_DefDomain
from petram.phys.em1d.em1d_vac import EM1D_Vac
from petram.phys.em1d.em1d_coldplasma import EM1D_ColdPlasma
from petram.phys.em1d.em1d_model import EM1D_DefBdry
from petram.phys.em1d.em1d_pec import EM1D_PEC
from petram.phys.em1d.em1d_cont import EM1D_Continuity
from petram.phys.em1d.em1d_e import EM1D_E
from petram.phys.em1d.em1d_port import EM1D_Port
from petram.phys.em1d.em1d_model import EM1D_DefPair
from petram.phys.aux_variable import AUX_Variable
from petram.mfem_model import MFEM_InitRoot
from petram.mfem_model import MFEM_PostProcessRoot
from petram.postprocess.dxp_model import DataExport
from petram.postprocess.pc_export import PointCloud
from petram.mfem_model import MFEM_SolverRoot
from petram.solver.solver_model import SolveStep
from petram.solver.std_solver_model import StdSolver
from petram.solver.mumps_model import MUMPS
from petram.solver.iterative_model import Iterative
from petram.solver.superlu_model import SuperLU
from collections import OrderedDict

def make_model():
    obj1 = MFEM_ModelRoot()
    obj1.root_path = '/tmp/piscope_shiraiwa/bastet01.pid3789450/.###ifigure_home_shiraiwa_src_TwoPiTest_piscope_projects_em1d_coldplasma.pfz/proj/model1/mfem'
    obj1.model_path = '/home/shiraiwa/src/TwoPiTest/piscope_projects'
    obj1.pkg_versions = {'petram_rf': '26.2.7', 'petram': '26.8.29', 'petram_extraphysnumba': '25.11.28', 'petram_rfp': '26.7.7', 'petram_extraphys': '25.11.28', 'petram_dpg': '26.6.29', 'petram_ds': '1.2.0', 'petram_whp': '25.11.28'}
    obj2 = obj1.add_node(name = "General", cls = MFEM_GeneralRoot)
    obj2.savegz = 'off'
    obj2.ns_name = "global"
    obj3 = obj1.add_node(name = "Geometry", cls = MFEM_GeomRoot)
    obj4 = obj1.add_node(name = "Mesh", cls = MFEM_MeshRoot)
    obj5 = obj4.add_node(name = "MeshGroup1", cls = MFEMMesh)
    obj6 = obj5.add_node(name = "Mesh1D1", cls = Mesh1D)
    obj6.length_txt = '0.5, 0.05'
    obj6.nsegs_txt = '600, 60'
    obj6.mesh_x0_txt = '-0.5'
    obj7 = obj1.add_node(name = "Phys", cls = MFEM_PhysRoot)
    obj8 = obj7.add_node(name = "EM1D1", cls = EM1D)
    obj8.dep_vars_suffix = '1'
    obj8.order_txt = 'order'
    obj8.freq_txt = 'freq'
    obj9 = obj8.add_node(name = "Domain", cls = EM1D_DefDomain)
    obj9.ky_txt = 0
    obj9.kz_txt = 0
    obj9.Einit_m_txt = '[0, 0, 0]'
    obj10 = obj9.add_node(name = "Vac1", cls = EM1D_Vac)
    obj10.enabled = False
    obj10.sel_index_txt = '2'
    obj10.epsilonr_txt = '(1+0j)'
    obj10.mur_txt = '(1+0j)'
    obj10.sigma_txt = '0.03'
    obj10.ky_txt = 'ky'
    obj10.kz_txt = 'kz'
    obj10.Einit_m_txt = '[0, 0, 0]'
    obj11 = obj9.add_node(name = "Vac2", cls = EM1D_Vac)
    obj11.sel_index_txt = '2'
    obj11.epsilonr_txt = '(1+0j)'
    obj11.mur_txt = '(1+0j)'
    obj11.sigma_txt = '0j'
    obj11.ky_txt = 'ky'
    obj11.kz_txt = 'kz'
    obj11.Einit_m_txt = '[0, 0, 0]'
    obj12 = obj9.add_node(name = "ColdPlasma1", cls = EM1D_ColdPlasma)
    obj12.sel_index_txt = '1'
    obj12.bext_txt = '=[0,0,Bnorm]'
    obj12.dens_e_txt = '=dens'
    obj12.temperature_txt = '500'
    obj12.dens_i_txt = '=dens,'
    obj12.mass_txt = '2, '
    obj12.charge_q_txt = '1, '
    obj12.ky_txt = 'ky'
    obj12.kz_txt = 'kz'
    obj12.Einit_m_txt = '[0, 0, 0]'
    obj13 = obj8.add_node(name = "Boundary", cls = EM1D_DefBdry)
    obj13.Einit_m_txt = '[0, 0, 0]'
    obj14 = obj13.add_node(name = "PEC1", cls = EM1D_PEC)
    obj14.sel_index_txt = '1'
    obj14.Einit_m_txt = '[0, 0, 0]'
    obj15 = obj13.add_node(name = "Continuity1", cls = EM1D_Continuity)
    obj15.sel_readonly = False
    obj15.sel_index_txt = '2'
    obj16 = obj13.add_node(name = "E1", cls = EM1D_E)
    obj16.enabled = False
    obj16.sel_index_txt = '1'
    obj16.E_z_txt = '1'
    obj16.E_m_txt = '[0, 0]'
    obj16.Einit_m_txt = '[0, 0, 0]'
    obj17 = obj13.add_node(name = "Port1", cls = EM1D_Port)
    obj17.sel_index_txt = '3'
    obj17.inc_amp_y_txt = '0'
    obj17.inc_amp_z_txt = '1'
    obj17.inc_amp_m_txt = '[1., 0.]'
    obj17.inc_phase_txt = '0.0'
    obj17.epsilonr_txt = '(1+0j)'
    obj17.mur_txt = '(1+0j)'
    obj17.ky_txt = 'ky'
    obj17.kz_txt = 'kz'
    obj17.Einit_m_txt = '[0, 0, 0]'
    obj17.port_idx = '1'
    obj18 = obj13.add_node(name = "Port2", cls = EM1D_Port)
    obj18.enabled = False
    obj18.sel_index_txt = '1'
    obj18.inc_amp_y_txt = '0.0'
    obj18.inc_amp_z_txt = '1.0'
    obj18.inc_amp_m_txt = '[1., 0.]'
    obj18.inc_phase_txt = '0.0'
    obj18.epsilonr_txt = '(1+0j)'
    obj18.mur_txt = '(1+0j)'
    obj18.ky_txt = 'ky'
    obj18.kz_txt = 'kz'
    obj18.Einit_m_txt = '[0, 0, 0]'
    obj18.isTimeDependent_RHS = False
    obj18.port_idx = '2'
    obj19 = obj8.add_node(name = "Pair", cls = EM1D_DefPair)
    obj20 = obj8.add_node(name = "Variable1", cls = AUX_Variable)
    obj20.enabled = False
    obj20.variable_name = 'Ez_0_4'
    obj20.aux_connection = OrderedDict({0: ('EM1D1', 2)})
    obj20.oprt_diag_txt = '-1'
    obj20.oprt1_0_txt = '=delta(0.4)'
    obj20.oprt1_0 = '=delta(0.4)'
    obj20.oprt2_0 = ''
    obj20.oprt2_0_txt = ''
    obj21 = obj1.add_node(name = "InitialValue", cls = MFEM_InitRoot)
    obj22 = obj1.add_node(name = "PostProcess", cls = MFEM_PostProcessRoot)
    obj23 = obj22.add_node(name = "DataExport1", cls = DataExport)
    obj23.use_scanner = False
    obj24 = obj23.add_node(name = "PointCloud1", cls = PointCloud)
    obj24.pc_x_txt = 'linspace(-0.5, 0.05, 200)'
    obj24.pc_y_txt = 'zeros(200)'
    obj24.pc_z_txt = 'zeros(200)'
    obj24.export_expr = 'E1z'
    obj25 = obj1.add_node(name = "Solver", cls = MFEM_SolverRoot)
    obj26 = obj25.add_node(name = "SolveStep1", cls = SolveStep)
    obj26.postprocess_sol = 'DataExport1'
    obj27 = obj26.add_node(name = "StdSolver1", cls = StdSolver)
    obj27.phys_model = 'EM1D1'
    obj28 = obj27.add_node(name = "MUMPS1", cls = MUMPS)
    obj28.enabled = False
    obj28.log_level = 1
    obj28.write_mat = True
    obj29 = obj27.add_node(name = "Iterative1", cls = Iterative)
    obj29.enabled = False
    obj29.log_level = 1
    obj29.maxiter = 2000
    obj29.kdim = 100
    obj29.preconditioners = [('E1x', ['None', 'GS']), ('E1y', ['None', 'GS']), ('E1z', ['None', 'GS']), ('Ey_port1', ['None', 'None']), ('Ez_port1', ['None', 'None']), ('Ey_port2', ['None', 'None']), ('Ez_port2', ['None', 'None'])]
    obj30 = obj27.add_node(name = "SuperLU1", cls = SuperLU)
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