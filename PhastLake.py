####################################################
#
# PhastLake.py - a lake boundary condition package
# for the USGS's PHAST reactive transport model
#
####################################################

from numpy import *
from pandas import *
from scipy import interpolate
from scipy.spatial import ConvexHull

options.mode.chained_assignment = None


class Lake:
    
    def __init__(self, gridFrame, grid):
        
        # moles of water per L of water
        mwH2O = 2.*1.0079 + 15.9994
        molesH2O = 1000. / mwH2O
        
        # lake geometry as implied by grid data frame
        cellArea = grid.dx * grid.dy
        self.area = gridFrame['lake'].sum() * cellArea      # surface area
        self.vol = gridFrame['depth'].sum() * cellArea      # water volume
        
        # water balance external constraints
        inFile = open('fluxes.txt','r')
        for line in inFile:
            lineInput = line.split()
            if lineInput[0] == 'Evap':
                self.evap = -float(lineInput[1]) * self.area    # evap rate
            elif lineInput[0] == 'Ppt':
                self.ppt = float(lineInput[1]) * self.area      # precip rate
            elif lineInput[0] == 'GWInflow':                
                self.gwInflow = float(lineInput[1])             # GW inflow (from PHAST)
            elif lineInput[0] == 'GWOutflow':                 
                self.gwOutflow  = -float(lineInput[1])          # GW outflow (from PHAST)             
            elif lineInput[0] == 'Inflow':                      # stream inflow (fixed or computed)
                self.inflow = float(lineInput[1])
                self.inflowBal = bool(int(lineInput[2]))                
            elif lineInput[0] == 'Outflow':                     # stream outflow (fixed or computed)
                self.outflow = float(lineInput[1])
                self.outflowBal = bool(int(lineInput[2]))                
        inFile.close()
        self.Balances()                                     # compute water balances
        self.fStream, self.fGW = self.EndMix()              # component fractions of end-member lake water
        self.evapConc = -(self.evap + self.ppt)/(self.inflow + self.gwInflow + self.ppt)  # evapo-concentratio
        self.molesEvap = self.evapConc * molesH2O           # represent evapo-concentration in PHREEQC/PHAST
        self.tHalf = self.HalfMix()                         # mixing half-life in lake at steady-state
        
    def Balances(self):
        # estimate unknown flow balance component (surface water inflow or outflow) by volume conservation
        if self.inflowBal:
            self.inflow = -self.outflow - self.evap - self.ppt - self.gwInflow - self.gwOutflow  
        else:
            self.outflow = -self.inflow - self.evap - self.ppt - self.gwInflow - self.gwOutflow  
   
    def EndMix(self):
        # compute end-state fractional composition of lake as blend of surface water and groundwater inflows
        totInflow = self.inflow + self.gwInflow
        fSurface = self.inflow/totInflow
        fGW = self.gwInflow/totInflow        
        return fSurface, fGW

    def HalfMix(self):
        # determine the lake's mixing half-life/blend rate coefficient
        halfLife = self.vol / (self.inflow + self.gwInflow)
        return halfLife


class Grid:
    
    def __init__(self, surface, zLake, lakePt):
        
        # create a uniform grid with supplied specs
        inFile = open('grid.txt','r')
        for line in inFile:
            lineInput = line.split()
            if lineInput[0] == 'x0':
                self.x0 = float(lineInput[1])
            elif lineInput[0] == 'xf':
                xf = float(lineInput[1])
            elif lineInput[0] == 'y0':
                self.y0 = float(lineInput[1])
            elif lineInput[0] == 'yf':
                yf = float(lineInput[1])                
            elif lineInput[0] == 'nx':
                self.nx = int(lineInput[1])                
            else:               # grid discretization along y-direction
                self.ny = int(lineInput[1])
        inFile.close()
        self.dx = (xf-self.x0)/self.nx
        self.dy = (yf-self.y0)/self.ny
        xGrid = arange(self.x0, xf, self.dx) + 0.5*self.dx
        yGrid = arange(self.y0, yf, self.dy) + 0.5*self.dy
        X, Y = meshgrid(xGrid,yGrid)
        self.x = X.flatten()
        self.y = Y.flatten()
        self.connects = self.Connections()      # convention preserves x & y ordering        
        
        # interpolator for elevation of lake bottom (bathymetry)
        xRef = array(surface['x'])
        yRef = array(surface['y'])
        zRef = array(surface['z'])        
        f = interpolate.LinearNDInterpolator((xRef, yRef), zRef)
        self.z = f(self.x, self.y)
        
        # set up initial assumption about lake extent (to seed iteration)
        self.submerged = (self.z < zLake)
        lakePtIndex = self.FindIndex(lakePt)
        self.lake = zeros(len(self.x), int)           
        self.lake[lakePtIndex] = 1

    def GridFrame(self):
        # package grid components as a data frame
        gridFrame = DataFrame(data = {'x':self.x, 'y':self.y, 'z':self.z, 'submerged':self.submerged, 'lake':self.lake})
        return gridFrame
       
    def FindIndex(self, pt):
        # return grid cell's index number
        col = int((pt[0] - self.x0) / self.dx)
        row = int((pt[1] - self.y0) / self.dy)        
        return row*self.nx + col

    def Connections(self):
        # assign connections to each grid cell for easy reference
        connects = []
        for i in range(self.nx*self.ny): connects.append([])
        # process horizontal connections
        for j in range(self.ny):
            for i in range(self.nx-1):
                connects[i + j*self.nx].append(i+1 + j*self.nx)
                connects[i+1 + j*self.nx].append(i + j*self.nx)
        # process vertical connections
        for i in range(self.nx):
            for j in range(self.ny-1):
                connects[i + j*self.nx].append(i + (j+1)*self.nx)
                connects[i + (j+1)*self.nx].append(i + j*self.nx)                
        return connects       


def PhastLake(zLake, lakePt):            # main script
    
    # process elevation data frame (land surface, without any surface water)
    surface = read_csv('FinalState.csv')
    
    # set up grid object; find starting point for lake cell connection iterative search    
    grid = Grid(surface, zLake, lakePt)
    gridFrame = grid.GridFrame()
    print('Initialized grid.')
    
    # iterative lake cell continuity checker
    maxIter = 100
    iterNum = 0
    numLakeCells = 0
    numLakeCells0 = 1
    print('Iterating ...')    
    while (iterNum < maxIter) and (numLakeCells != numLakeCells0):
        numLakeCells0 = numLakeCells
        lakeCells = list(gridFrame.index[gridFrame['lake']==1])        # current list of lake cell indices
        spread = []
        for center in lakeCells: spread.extend(grid.connects[center])   # connecting cells to lake cells
        spread = list(set(spread) - set(lakeCells))      # connecting cells to lake cells that are not marked as lake cells
        gridFrame['lake'].iloc[spread] = 1*(gridFrame['submerged'].iloc[spread]==True) + 0*(gridFrame['submerged'].iloc[spread]==False)
        numLakeCells = gridFrame['lake'].sum()
        iterNum += 1    

    # finish processing grid data frame
    gridFrame['depth'] = (zLake-gridFrame['z'])*gridFrame['lake']
    gridFrame.to_csv('lakeGrid.csv', index=False)

    # write convex hull of points
    lakePoints = gridFrame[['x', 'y']][gridFrame['lake']==1].values
    hull = ConvexHull(lakePoints)
    xHull = transpose(lakePoints)[0][hull.vertices]
    yHull = transpose(lakePoints)[1][hull.vertices]
    lakeHull = DataFrame(data={'x':xHull, 'y':yHull})
    lakeHull.to_csv('lakeHull.csv', index=False)
    print('Wrote convex hull of lake points to file.\n')

    # create lake object
    lake = Lake(gridFrame, grid)
    print('Lake water flux balances:')
    print('\tEvaporation = ', lake.evap)
    print('\tPrecipitation = ', lake.ppt)    
    print('\tGroundwater influx = ', lake.gwInflow)
    print('\tGroundwater outflux = ', lake.gwOutflow)    
    print('\tSurface water influx = ', lake.inflow)
    print('\tSurface water outflux = ', lake.outflow, '\n')     
    print('Solute concentration in lake:')
    print('\tStream mixing fraction = ', lake.fStream)
    print('\tGroundwater mixing fraction = ', lake.fGW)    
    print('\tEvapo-concentration factor = ', lake.evapConc)
    print('\tMoles evaporated from 1 L = ', lake.molesEvap)    
    print('\tMixing half-life = ', lake.tHalf)

    print(' ')
    print('Done.')

 
### run script ###
zLake = 1110.       # zLake = lake surface elevation at steady-state
lakePt = [3671.6417910448, 5582.0895522388]     # location of a point within lake (to seed iteration)
PhastLake(zLake, lakePt)