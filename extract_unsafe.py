import pickle
import glob
from StateSpacePartitioner import StateSpacePartitioner
from Workspace import *
import matplotlib.pyplot as plt

pts = [[1.000000,1.500000],
[1.054214,1.545948],
[1.106700,1.592276],
[1.157464,1.638978],
[1.206512,1.686052],
[1.253850,1.733493],
[1.299485,1.781297],
[1.344633,1.828504],
[1.389489,1.874970],
[1.434052,1.920711],
[1.478316,1.965743],
[1.522281,2.010080],
[1.565638,2.053502],
[1.607875,2.095629],
[1.649020,2.136499],
[1.689101,2.176151],
[1.728146,2.214621],
[1.766181,2.251946],
[1.803231,2.288160],
[1.839322,2.323296],
[1.874478,2.357387],
[1.908723,2.390465],
[1.942080,2.422560],
[1.974573,2.453702],
[2.006223,2.483919],
[2.037305,2.513715],
[2.067634,2.544144],
[2.097226,2.575203],
[2.126096,2.606892],
[2.154260,2.639210],
[2.181731,2.672155],
[2.208775,2.705895],
[2.235638,2.740594],
[2.262324,2.776260],
[2.290175,2.812382],
[2.320070,2.848649],
[2.352071,2.885077],
[2.384015,2.921805],
[2.415804,2.958838],
[2.446946,2.996601],
[2.479016,3.037410],
[2.514102,3.077378],
[2.552158,3.116524],
[2.592944,3.155027],
[2.640668,3.173566],
[2.688324,3.191570],
[2.735914,3.209058],
[2.783438,3.226047],
[2.830900,3.242556],
[2.878300,3.258601],
[2.925640,3.274197],
[2.972922,3.289362],
[3.020147,3.304109],
[3.067316,3.318454],
[3.114432,3.332410],
[3.161495,3.345991],
[3.208506,3.359210],
[3.255468,3.372080],
[3.302380,3.384612],
[3.349245,3.396819],
[3.396063,3.408712],
[3.442836,3.420301],
[3.489815,3.425385],
[3.536775,3.430302],
[3.583719,3.435058],
[3.630645,3.439658],
[3.677556,3.444108],
[3.724450,3.448413],
[3.771330,3.452576],
[3.818194,3.456603],
[3.865045,3.460498],
[3.911881,3.464266],
[3.958704,3.467911],
[4.005515,3.471436],
[4.013685,3.492518],
[4.022747,3.512920],
[4.032974,3.532546],
[4.044329,3.551486],
[4.052519,3.602534],
[4.061005,3.654254],
[4.069775,3.706656],
[4.078816,3.759749],
[4.089879,3.812789],
[4.103129,3.865637],
[4.118614,3.918197],
[4.135518,3.969155],
[4.153093,4.017414],
[4.171305,4.062997],
[4.190122,4.105930],
[4.208574,4.150619],
[4.226465,4.198068],
[4.243883,4.248077],
[4.261886,4.297701],
[4.280469,4.346950],
[4.299629,4.395833],
[4.319361,4.444360],
[4.339662,4.492540],
[4.360529,4.540383],
[4.381958,4.587897],
[4.403946,4.635092],
[4.426489,4.681976]]

eps = 1E-3
def in_region(regions,x):
    
    ret = []
    for idx,region in enumerate(regions):
        H = region['PolygonH']['A']
        b = region['PolygonH']['b']
        diff = H.dot(x.reshape((2,1))) -b
        if  (diff<=eps).all():
            print(idx) 
            ret.append(idx)
    return ret

num_obstacles = 3
num_lasers = 8
workspace = Workspace(8,num_lasers,'obstacles.json')
num_integrators = 1

higher_deriv_bound = 1.0
grid_size = [0.05,1.0]
neighborhood_radius = 0.01
    


regions_passed = []

dir = 'NN20_grid_005/'

with open(dir +'symbolic_states','rb') as f:
    partitioner = pickle.load(f)
# partitioner = StateSpacePartitioner(workspace, num_integrators, higher_deriv_bound, grid_size, neighborhood_radius)
# partitioner.partition()
for pt in pts:
    rg = in_region(partitioner.symbolic_states,np.array(pt))
    regions_passed.append(rg)
traj = [region for regions in regions_passed for region in regions]
graph_files = glob.glob(dir + '/graph_*')
unsafe_files = glob.glob(dir + '/safety_*')
graph_files.sort()
unsafe_files.sort()

states = set(range(len(partitioner.symbolic_states)))
AREA = 36
safe_ratio = []
for i in range(len(unsafe_files)):
    if(i < len(unsafe_files)-1):
        continue
    safe_area = 0
    with open(unsafe_files[i],'rb') as f:
        unsafe = set(pickle.load(f))
        safe = states - unsafe
        for region_idx in safe: 
            safe_area += partitioner.symbolic_states[region_idx]['Polygon'].area
        safe_ratio.append(100*safe_area / AREA)

    if(i == len(unsafe_files) - 1): 
        for idx,candidates in enumerate(regions_passed):
            for reg in candidates:
                if(reg in unsafe):
                    print(idx,reg)

        # partitioner.plotWorkspacePartitions(safe, show  = True, traj = traj)

    if(i>=29):
        # partitioner.plotWorkspacePartitions(safe, show  = True, save_file = dir + 'results/epoch_%d.png'%(i+1), traj = traj)
        partitioner.plotWorkspacePartitions(partitioner.obstacle_symbolic_states + list(unsafe), show  = True, save_file = dir + 'results/epoch_%d.png'%(i+1), traj = traj)

print(safe_ratio)

ax = plt.subplot(111)
ax.plot(range(1,len(safe_ratio) +1), safe_ratio)
ax.set_xlabel('Epoch')
ax.set_ylabel('% Area')
ax.set_ylim(ymin=12)
plt.show()
pass 

