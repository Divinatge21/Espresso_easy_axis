import numpy as np
import espressomd
import espressomd.io.writer.vtf
import matplotlib.pyplot as plt
from espressomd import thermostat
import vtk
import os
from collections import defaultdict
from espressomd.interactions import RigidBond
import tqdm
from espressomd.observables import MagneticDipoleMoment
import espressomd.magnetostatics as magnetostatics
import espressomd.magnetostatics
from espressomd.magnetostatics import DipolarP3M





# параметр Ланжевена
ALPHA_P = 5.0

lj_sigma=1.0
lj_epsilon=1.0
lj_cut = 2.**(1./6.) * lj_sigma

# magnetic field constant
mu_0 = 1.0

# Particles
N = 1

# Volume fraction
# phi = rho * 4. / 3. * np.pi * ( lj_sigma / 2 )**3.
phi = 0.1/100.

rtype = 0  # Тип частицы

density = 6 *10 / np.pi * phi / lj_sigma**3
print(f'rho = {density:.4f}')

# Dipolar interaction parameter lambda = mu_0 m^2 /(4 pi sigma^3 kT)
dip_lambda = 1.0

# Temperature
kT =1. / 10000.

# Friction coefficient
gamma=1.0

# Time step
dt =0.00025

# box size 3d
box_size = (N * np.pi * 4./3. * (lj_sigma / 2.)**3. / phi)**(1./3.)

system = espressomd.System(box_l=(box_size,box_size,box_size)) 
system.time_step = dt
system.periodicity = [1, 1, 1]
np.random.seed(42)
system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=1)

# Lennard-Jones interaction
system.non_bonded_inter[0,0].lennard_jones.set_params(epsilon=lj_epsilon,sigma=lj_sigma,cutoff=lj_cut, shift="auto")

print("Simulate {} particles in a cubic simulation box of length {} at density {}."
      .format(N, box_size, density).strip())
      
# Random dipole moments
np.random.seed(seed = 512)
dip_zeta = np.random.random((N,1)) *2. * np.pi
dip_cos_omega = 2*np.random.random((N,1)) -1
dip_sin_omega = np.sin(np.arccos(dip_cos_omega))
dip = np.hstack((
   dip_sin_omega *np.cos(dip_zeta),
   dip_sin_omega *np.sin(dip_zeta),
   dip_cos_omega))
   
np.random.seed(seed = 512)
easy_psi = np.random.random((N,1)) *2. * np.pi
easy_cos_xsi = 2*np.random.random((N,1)) -1
easy_sin_xsi = np.sin(np.arccos(easy_cos_xsi))
easy = np.hstack((
   easy_sin_xsi *np.cos(easy_psi),
   easy_sin_xsi *np.sin(easy_psi),
   easy_cos_xsi))


# Add particles
# ось легкого намагничивания
sigma_ani = 10.0

# dipole moment
dipm = np.sqrt(dip_lambda*4*np.pi*lj_sigma**3.*kT / mu_0)
print("dipm = {}".format(dipm))

chi_L = 8. * dip_lambda * phi
print('chi_L = %.4f' % chi_L)
print('sigma_ani = %.4f' % sigma_ani)



#Добавления частиц
for i in range(N):
    system.part.add(id=i, type=rtype, pos = np.random.random(3), fix =[1, 1, 1],
                    rotation=[0, 0, 0], dip=[1, 1, 1], 
                    easy_axis = [1, 1, 1], sigma_m = 1)

# Remove overlap between particles by means of the steepest descent method
#system.integrator.set_steepest_descent(
 #   f_max=0,gamma=0.1,max_displacement=0.05)

particle_ids = [p.id for p in system.part]

dipm_tot_calc = MagneticDipoleMoment(ids = particle_ids)
magnetization0 =[]
magn_temp = 0
magn_temp += dipm_tot_calc.calculate()[2]
magnetization0.append(magn_temp / N)
print("magnetizations = {}".format(magnetization0))

#while system.analysis.energy()["total"] > 5*kT*N:
 #   system.integrator.run(20)

# Switch to velocity Verlet integrator
system.integrator.set_vv()

# tune verlet list skin
system.cell_system.skin = 0.8

# Setup dipolar P3M
#accuracy = 5E-4
#system.actors.add(DipolarP3M(accuracy=accuracy,prefactor=dip_lambda*lj_sigma**3*kT))

dipm_tot_calc = MagneticDipoleMoment(ids = particle_ids)

# Sampling
loops = 1000

print('Sampling ...')
# calculate initial total dipole moment
dipm_tot_temp = dipm_tot_calc.calculate()

# dipole moment
dipm = np.sqrt(dip_lambda*4*np.pi*lj_sigma**3.*kT / mu_0)
print("dipm = {}".format(dipm))



alphas = [5.0]
print("alphaP = {}".format(alphas))
print("Sigma_any = {}".format(sigma_ani))
print("Lambda = {}".format(dip_lambda))
# remove all constraints
system.constraints.clear()

# list of magnetization in field direction
magnetization = []

#запись файла vtf
output_file_path = "dip_trajectory_ferro_ap5.vtf"
fp = open(output_file_path, mode='w+t')

# Запись блока с структурой
espressomd.io.writer.vtf.writevsf(system, fp)

# Запись начальных позиций
espressomd.io.writer.vtf.writevcf(system, fp)

def write_vtk_file(s, fname, time, real_part_ids, cluster_marker, bonds=[]):
    # Рассчет энергии и добавление к имени файла
    total_energy = s.analysis.energy()['total']
    energies.append(total_energy)
    
    fname = fname + '.step.{}.vtk'.format(time)
    point_num = len(s.part[:])
    part_idx = list(np.arange(point_num))
    
    if len(real_part_ids) > 0:
        part_idx = real_part_ids
        point_num = len(real_part_ids)
    
    ugrid_l = vtk.vtkUnstructuredGrid()
    
    # Transfer points to vtk structure
    points = vtk.vtkPoints()
    for i in part_idx:
        px, py, pz = s.part[i].pos
        points.InsertNextPoint(px, py, pz)
    ugrid_l.SetPoints(points)
    
    point_data = ugrid_l.GetPointData()
    
    # Add magnetic moments
    moments = vtk.vtkDoubleArray()
    moments.SetNumberOfComponents(3)
    moments.SetName('mag_moments')
    for i in part_idx:
        mx, my, mz = s.part[i].dip
        moments.InsertNextTuple3(mx, my, mz) 
    point_data.AddArray(moments)

    # Add easy axes
    easy_axes = vtk.vtkDoubleArray()
    easy_axes.SetNumberOfComponents(3)
    easy_axes.SetName('easy_axes')
    for i in part_idx:
        ex, ey, ez = s.part[i].easy_axis
        easy_axes.InsertNextTuple3(ex, ey, ez)
    point_data.AddArray(easy_axes)
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(ugrid_l)
    writer.Write()

positions_accumulator = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
moments_accumulator = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
easy_accumulator = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
counts = defaultdict(int)

magnetization_z =[]
magnetization_y =[]
magnetization_x =[]


# number of loops for sampling
loops_m = 2000

# Основной цикл моделирования

for alpha in alphas:
    # Включаем магнитное поле
    print("Sample for alpha = {}".format(alpha))
    #H_dipm = (alpha * kT)
    #H_field = [0, 0, H_dipm * np.cos(w)]
    print("Set magnetic field constraint...")
    #H_constraint = espressomd.constraints.HomogeneousMagneticField(H=H_field)
    #system.constraints.add(H_constraint)
    print("done\n")

    print("Sampling...")
    total_energy = 0
    loop_steps = []
    energies = []
    vtk_files = []
    magn_temp_z = 0
    magn_temp_y = 0
    magn_temp_x = 0
    
    for i in range(loops_m):
        system.integrator.run(1)
        loop_steps.append(i + 1)
        energies.append(system.analysis.energy()['total'])
        total_energy += energies[0]

        magnetic_moment = dipm_tot_calc.calculate() # This line MUST be inside the loop

        magn_temp_z += magnetic_moment[2]
        magn_temp_y += magnetic_moment[1]
        magn_temp_x += magnetic_moment[0]
        
    

        print("progress: {:3.0f}%".format((i + 1) * 100. / loops_m), end="\r")
    print("\n")
 
    average_energy = sum(energies) / len(energies)
    print("\nAverage energy for alpha_P = {}: {}".format(alpha, average_energy))
    
    filename = 'Dip_Dip_ferro_sigma_{}_alphaP_{}_chi_{}_E_{:.2f}.vtk'.format(sigma_ani, alpha, chi_L,average_energy)
    #write_vtk_file(system, filename, loops_m, range(N), [0] * N)
    
    
    print("Magnetization...")
    # save average magnetization
     
    magnetization_z.append(magn_temp_z / loops_m / N)
    magnetization_y.append(magn_temp_y / loops_m / N)
    magnetization_x.append(magn_temp_x / loops_m / N)
    
    print("Sampling for alpha = {} done \n".format(alpha))
    print("magnetizations_x  = {}".format(magnetization_x))
    print("magnetizations_y  = {}".format(magnetization_y))
    print("magnetizations_z  = {}".format(magnetization_z))
    
    print("total progress: {:5.1f}%\n".format((alphas.index(alpha)+1)*100./len(alphas)))
    # remove constraint
    system.constraints.clear()
print("Magnetization curve sampling done")

for particle in system.part:
   print("Магнитный момент обездвиженной частицы {}: {}".format(particle.id, particle.dip))
   print("Ось легкого намагничивания обездвиженной частицы {}: {}".format(particle.id, particle.easy_axis))
   print("Текущее положение частицы {}: {}".format(particle.id, particle.pos))
   



lj_cap = 0
system.force_cap = lj_cap
print(system.non_bonded_inter[0, 0].lennard_jones)









fp.close()