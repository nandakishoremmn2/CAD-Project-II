import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle, time
from optparse import OptionParser

class Flux(object):
	def __init__(self, shape):
		self.d = np.zeros(shape)		
		self.u = np.zeros(shape)		
		self.v = np.zeros(shape)		
		self.p = np.zeros(shape)		

class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def normal(self):
		return Point(-self.y, self.x)

	def __len__(self):
		return np.sqrt(self.x**2 + self.y**2)

	def __add__(self, other):
		return Point(self.x+other.x, self.y+other.y)

	def __sub__(self, other):
		return Point(self.x-other.x, self.y-other.y)

	def __div__(self, other):
		return Point(self.x/other, self.y/other)

	def __neg__(self):
		return Point(-self.x, -self.y)

	def __repr__(self):
		return "("+str(self.x)+","+str(self.y)+")"

class Solver(object):
	def __init__(self, *args, **kwargs):
		self.start = time.time()
		self.plotInterval = kwargs["plotInterval"]
		self.tolerance = kwargs["tolerance"]
		self.dumpfile = kwargs["savefile"]
		self.plt = plt
		self.fig = plt.figure()
		self.trim = np.vectorize( lambda x: x if x>0. else 0. )
		self.beta_eq = np.vectorize( lambda x: 0. if abs(x)>=1 else -1. )
		self.alpha_eq = np.vectorize( lambda x: 0.5*(1. + (1. if x>=0 else -1.) ) )
		self.residue = []

		self.readMesh(kwargs["meshfile"])
		self.findNormals()
		self.findVolumes()
		self.setParameters()
		self.intiVariables()
		self.loadData() if kwargs["resume"] else self.setField()
		self.flux = Flux(self.p.shape)
		self.primitive_to_conservative()
		self.marchOneStep()
		self.findResidue()

	def readMesh(self, meshfile):
		data 		= open(meshfile)
		self.size 	= Point(*map(int, data.readline().split()))

		self.mesh 	= np.array([Point(*map(float, coords.split())) for coords in data], dtype=Point).reshape(self.size.y, self.size.x)
		self.mesh 	= np.insert(self.mesh, 0, self.mesh[:,0]+self.mesh[:,0]-self.mesh[:,1], axis=1 )
		self.mesh 	= np.insert(self.mesh, -1, self.mesh[:,-1]+self.mesh[:,-1]-self.mesh[:,-2], axis=1 )
		# self.flatten()
		self.size 	= Point(*self.mesh.shape[::-1])
		self.x 		= np.array(np.vectorize(lambda a:a.x)(self.mesh[:-1,:-1] + self.mesh[1:,1:] + self.mesh[1:,:-1] + self.mesh[:-1,1:]), dtype=float)/4
		self.y 		= np.array(np.vectorize(lambda a:a.y)(self.mesh[:-1,:-1] + self.mesh[1:,1:] + self.mesh[1:,:-1] + self.mesh[:-1,1:]), dtype=float)/4

	def flatten(self):
		for j in xrange(len(self.mesh[:,0])):
			for i in xrange(len(self.mesh[j,:])):
				self.mesh[j,i].y = self.mesh[j,0].y

	def findNormals(self):
		self.xnx, self.xny, self.xar = [n.transpose() for n in self.getNormals(self.mesh.transpose())]
		self.ynx, self.yny, self.yar = [-n for n in self.getNormals(self.mesh)]
		self.yar = -self.yar

	def getNormals(self, mesh):
		normalsX = np.array([ [n.normal().x for n in coords[:-1]-coords[1:]] for coords in mesh ])
		normalsY = np.array([ [n.normal().y for n in coords[:-1]-coords[1:]] for coords in mesh ])
		areas    = np.sqrt( normalsX**2 + normalsY**2 )
		return normalsX/areas, normalsY/areas, areas

	def findVolumes(self):
		self.volume = np.vectorize(self.getVolume)(self.mesh[:-1,:-1], self.mesh[1:,:-1], self.mesh[1:,1:], self.mesh[:-1,1:])

	def getVolume(self, r1, r2, r3, r4):
		v1 = .5*abs( (r2.x-r1.x)*(r3.y-r2.y) - (r3.x-r2.x)*(r2.y-r1.y) )
		v2 = .5*abs( (r3.x-r2.x)*(r4.y-r3.y) - (r4.x-r3.x)*(r3.y-r2.y) )
		return v1+v2

	def setParameters(self):
		self.c 			= 1.
		self.gamma		= 1.4
		self.MachNo 	= .4
		self.p0 		= 100000.
		self.d0 		= 1.225
		self.u0 		= self.MachNo*np.sqrt(self.gamma*self.p0/self.d0)
		self.v0 		= 0.
		self.cfl 		= .4
		self.beta 		= 10.
		self.timeStep 	= .000001
		self.iterations = 0

	def intiVariables(self):
		self.xFlux = Flux(self.xnx.shape)
		self.yFlux = Flux(self.ynx.shape)

	def findResidue(self):
		RN0 = \
			( self.flux.p[1:-1,3:-3]/(self.u0**3*self.d0) )**2 +\
			( self.flux.u[1:-1,3:-3]/(self.u0**2*self.d0) )**2 +\
			( self.flux.v[1:-1,3:-3]/(self.u0**2*self.d0) )**2 +\
			( self.flux.d[1:-1,3:-3]/(self.d0*self.u0) )**2 
		self.residue.append( np.sqrt(np.sum(RN0)) )

	def findFlux(self):
		self.h 				= self.p*self.gamma/self.d/(self.gamma-1) + .5*(self.u**2+self.v**2)
		self.a 				= np.sqrt(np.abs(self.p/self.d*self.gamma))

		self.xMPos 			= ( self.u[:,:-1]*self.xnx[:,1:-1] + self.v[:,:-1]*self.xny[:,1:-1] )/( self.a[:,:-1] + self.a[:,1:] )*2.
		self.xMNeg 			= ( self.u[:,1:]*self.xnx[:,1:-1] + self.v[:,1:]*self.xny[:,1:-1] )/( self.a[:,:-1] + self.a[:,1:] )*2.

		self.xAlpha_pos 	= self.alpha_eq(self.xMPos)
		self.xAlpha_neg 	= self.alpha_eq(-self.xMNeg)
		self.xBeta_pos 		= self.beta_eq(self.xMPos)
		self.xBeta_neg 		= self.beta_eq(self.xMNeg)
		self.xC_plus 		= self.xAlpha_pos*(1+self.xBeta_pos)*self.xMPos - .25*self.xBeta_pos*(1+self.xMPos)**2
		self.xC_minus 		= self.xAlpha_neg*(1+self.xBeta_neg)*self.xMNeg + .25*self.xBeta_neg*(1-self.xMNeg)**2
		self.xD_plus 		= .25*(1+self.xMPos)**2 * (2-self.xMPos)
		self.xD_minus 		= .25*(1-self.xMNeg)**2 * (2+self.xMNeg)
		self.xD_bar_plus 	= self.xAlpha_pos*(1+self.xBeta_pos) - self.xBeta_pos*self.xD_plus
		self.xD_bar_minus 	= self.xAlpha_pos*(1+self.xBeta_neg) - self.xBeta_neg*self.xD_minus

		self.xFlux.d[:,1:-1] = self.xar[:,1:-1]*(  self.d[:,:-1]*self.a[:,:-1]*self.xC_plus + self.d[:,1:]*self.a[:,1:]*self.xC_minus  )
		self.xFlux.u[:,1:-1] = self.xar[:,1:-1]*(  self.d[:,:-1]*self.a[:,:-1]*self.xC_plus*self.u[:,:-1] + self.d[:,1:]*self.a[:,1:]*self.xC_minus*self.u[:,1:]  ) +\
			self.xar[:,1:-1]*( self.xD_bar_plus*self.p[:,:-1] + self.xD_bar_minus*self.p[:,1:] ) * self.xnx[:,1:-1]
		self.xFlux.v[:,1:-1] = self.xar[:,1:-1]*(  self.d[:,:-1]*self.a[:,:-1]*self.xC_plus*self.v[:,:-1] + self.d[:,1:]*self.a[:,1:]*self.xC_minus*self.v[:,1:]  ) +\
			self.xar[:,1:-1]*( self.xD_bar_plus*self.p[:,:-1] + self.xD_bar_minus*self.p[:,1:] ) * self.xny[:,1:-1]
		self.xFlux.p[:,1:-1] = self.xar[:,1:-1]*(  self.d[:,:-1]*self.a[:,:-1]*self.xC_plus*self.h[:,:-1] + self.d[:,1:]*self.a[:,1:]*self.xC_minus*self.h[:,1:]  )

		self.yMPos 			= ( self.u[:-1,:]*self.ynx[1:-1,:] + self.v[:-1,:]*self.yny[1:-1,:] )/( self.a[:-1,:] + self.a[1:,:] )*2.
		self.yMNeg 			= ( self.u[1:,:]*self.ynx[1:-1,:] + self.v[1:,:]*self.yny[1:-1,:] )/( self.a[:-1,:] + self.a[1:,:] )*2.

		self.yAlpha_pos 	= self.alpha_eq(self.yMPos)
		self.yAlpha_neg 	= self.alpha_eq(-self.yMNeg)
		self.yBeta_pos 		= self.beta_eq(self.yMPos)
		self.yBeta_neg 		= self.beta_eq(self.yMNeg)
		self.yC_plus 		= self.yAlpha_pos*(1+self.yBeta_pos)*self.yMPos - .25*self.yBeta_pos*(1+self.yMPos)**2
		self.yC_minus 		= self.yAlpha_neg*(1+self.yBeta_neg)*self.yMNeg + .25*self.yBeta_neg*(1-self.yMNeg)**2
		self.yD_plus 		= .25*(1+self.yMPos)**2 * (2-self.yMPos)
		self.yD_minus 		= .25*(1-self.yMNeg)**2 * (2+self.yMNeg)
		self.yD_bar_plus 	= self.yAlpha_pos*(1+self.yBeta_pos) - self.yBeta_pos*self.yD_plus
		self.yD_bar_minus 	= self.yAlpha_pos*(1+self.yBeta_neg) - self.yBeta_neg*self.yD_minus

		self.yFlux.d[1:-1,:] = self.yar[1:-1,:]*(  self.d[:-1,:]*self.a[:-1,:]*self.yC_plus + self.d[1:,:]*self.a[1:,:]*self.yC_minus  )
		self.yFlux.u[1:-1,:] = self.yar[1:-1,:]*(  self.d[:-1,:]*self.a[:-1,:]*self.yC_plus*self.u[:-1,:] + self.d[1:,:]*self.a[1:,:]*self.yC_minus*self.u[1:,:]  ) +\
			self.yar[1:-1,:]*( self.yD_bar_plus*self.p[:-1,:] + self.yD_bar_minus*self.p[1:,:] ) * self.ynx[1:-1,:]
		self.yFlux.v[1:-1,:] = self.yar[1:-1,:]*(  self.d[:-1,:]*self.a[:-1,:]*self.yC_plus*self.v[:-1,:] + self.d[1:,:]*self.a[1:,:]*self.yC_minus*self.v[1:,:]  ) +\
			self.yar[1:-1,:]*( self.yD_bar_plus*self.p[:-1,:] + self.yD_bar_minus*self.p[1:,:] ) * self.yny[1:-1,:]
		self.yFlux.p[1:-1,:] = self.yar[1:-1,:]*(  self.d[:-1,:]*self.a[:-1,:]*self.yC_plus*self.h[:-1,:] + self.d[1:,:]*self.a[1:,:]*self.yC_minus*self.h[1:,:]  )

		self.findBoundaryFlux()

	def findBoundaryFlux(self):
		self.yFlux.d[0,:] = 0.
		self.yFlux.u[0,:] = self.p[0,:]*self.ynx[0,:]*self.yar[0,:]
		self.yFlux.v[0,:] = self.p[0,:]*self.yny[0,:]*self.yar[0,:]
		self.yFlux.p[0,:] = 0.

		self.yFlux.d[-1,:] = 0.
		self.yFlux.u[-1,:] = self.p[-1,:]*self.ynx[-1,:]*self.yar[-1,:]
		self.yFlux.v[-1,:] = self.p[-1,:]*self.yny[-1,:]*self.yar[-1,:]
		self.yFlux.p[-1,:] = 0.

	def findNetFlux(self):
		self.setBoundaryValues()
		self.primitive_to_conservative()
		self.findFlux()
		self.findBoundaryFlux()

		self.flux.d = self.xFlux.d[:,1:] - self.xFlux.d[:,:-1] + self.yFlux.d[1:,:] - self.yFlux.d[:-1,]
		self.flux.u = self.xFlux.u[:,1:] - self.xFlux.u[:,:-1] + self.yFlux.u[1:,:] - self.yFlux.u[:-1,]
		self.flux.v = self.xFlux.v[:,1:] - self.xFlux.v[:,:-1] + self.yFlux.v[1:,:] - self.yFlux.v[:-1,]
		self.flux.p = self.xFlux.p[:,1:] - self.xFlux.p[:,:-1] + self.yFlux.p[1:,:] - self.yFlux.p[:-1,]

		# self.setBoundaryValues()

	def setField(self):
		self.d = np.ones([self.size.y-1, self.size.x-1], dtype=float)*self.d0
		self.u = np.ones([self.size.y-1, self.size.x-1], dtype=float)*(self.u0)
		# self.u = np.ones([self.size.y-1, self.size.x-1], dtype=float)*(self.u0*((self.x-3.5)/-7.))
		# self.u = np.where(self.u<self.u0/2, self.u0/2, self.u)
		self.v = np.ones([self.size.y-1, self.size.x-1], dtype=float)*(self.v0+10)
		self.p = np.ones([self.size.y-1, self.size.x-1], dtype=float)*self.p0

		self.primitive_to_conservative()


	def primitive_to_conservative(self):
		self.du = self.d*self.u
		self.dv = self.d*self.v
		self.de = (self.gamma-1)*self.p + .5*self.d*(self.u**2+self.v**2)

	def conservative_to_primitive(self):
		self.u = self.du/self.d
		self.v = self.dv/self.d
		self.p = ( self.de - .5*self.d*( self.u**2+self.v**2 ) )/( self.gamma-1 )

	def setBoundaryValues(self):
		a = np.sqrt(self.gamma*self.p[:,0:2]/self.d[:,0:2])
		a = (a[:,0] + a[:,1])/2.
		self.inflowM = self.u[:,1]/a

		self.d[:,0] = np.where(self.inflowM>=1, self.d0, self.d0)
		self.u[:,0] = np.where(self.inflowM>=1, self.u0, self.u[:,1])
		self.v[:,0] = np.where(self.inflowM>=1, self.v0, self.v0)
		self.p[:,0] = np.where(self.inflowM>=1, self.p0, self.p0)

		a = np.sqrt(self.gamma*self.p[:,0:2]/self.d[:,0:2])
		a = (a[:,0] + a[:,1])/2.
		self.outflowM = self.u[:,1]/a

		self.d[:,-1] = np.where(self.outflowM>=1, self.d[:,-2], self.d[:,-2])
		self.u[:,-1] = np.where(self.outflowM>=1, self.u[:,-2], self.u0)
		self.v[:,-1] = np.where(self.outflowM>=1, self.v[:,-2], self.v[:,-2])
		self.p[:,-1] = np.where(self.outflowM>=1, self.p[:,-2], self.p[:,-2])

	def findLocalTimeStep(self):
		temp  	= np.abs(self.u*self.xnx[:,:-1]+self.v*self.xny[:,:-1])\
		 		+ np.abs(self.u*self.xnx[:,1:]+self.v*self.xny[:,1:])\
		 		+ np.abs(self.u*self.ynx[:-1,:]+self.v*self.yny[:-1,:])\
				+ np.abs(self.u*self.ynx[1:,:]+self.v*self.yny[1:,:])\
				+ 4*self.a

		self.timeStep = 4*self.volume*self.cfl/temp

		# self.timeStep = np.min(self.timeStep)

	def marchOneStep(self, *args):
		# self.setBoundaryValues()
		self.conservative_to_primitive()
		self.findNetFlux()
		self.findLocalTimeStep()

		self.primitive_to_conservative()

		self.d  = self.d  - self.timeStep*( self.flux.d )/self.volume
		self.du = self.du - self.timeStep*( self.flux.u )/self.volume
		self.dv = self.dv - self.timeStep*( self.flux.v )/self.volume
		self.de = self.de - self.timeStep*( self.flux.p )/self.volume

		# self.conservative_to_primitive()

		# self.setBoundaryValues()
		self.findResidue()


		if np.abs(self.residue[-1]/self.residue[0])<self.tolerance:
			return False

		self.iterations += 1
		return self.iterations

	def log(self):
		print "\tResidue = %.4g"%(np.abs(self.residue[-1]/self.residue[0])),
		print "\tMin. Time step = %.4g"%np.min(self.timeStep),
		print "\t Iterations = %d"%self.iterations,
		print "\tTime elapsed = %.2fs"%(time.time()-self.start),
		print ""

	def saveData(self):
		dumpfile 		= open(self.dumpfile, "w")
		pickle.dump(self.p, dumpfile)
		pickle.dump(self.u, dumpfile)
		pickle.dump(self.v, dumpfile)
		pickle.dump(self.h, dumpfile)
		pickle.dump(self.residue, dumpfile)
		pickle.dump(self.iterations, dumpfile)
		dumpfile.close()

	def saveAll(self):
		scipy.io.savemat("all_data.mat",
			{
				"x": self.x,
				"y": self.y,
				"p": self.p,
				"u": self.u,
				"v": self.v,
				"d": self.d,
				"iterations": self.iterations,
				"residue": self.residue
			})

	def loadData(self):
		dumpfile 		= open(self.dumpfile)
		self.d 			= pickle.load(dumpfile)
		self.u 			= pickle.load(dumpfile)
		self.v 			= pickle.load(dumpfile)
		self.h 			= pickle.load(dumpfile)
		self.residue 	= pickle.load(dumpfile)
		self.iterations = pickle.load(dumpfile)

	def plotMesh(self, ):
		plt.hold(True)
		for x, y in [zip(*[(coord.x, coord.y) for coord in coords]) for coords in self.mesh.transpose()]:
			self.plt.plot(x, y, 'b')
		for x, y in [zip(*[(coord.x, coord.y) for coord in coords]) for coords in self.mesh]:
			self.plt.plot(x, y, 'b')
		self.plt.axis("equal")
		self.plt.show()

	def plotField(self):
		# self.CS = self.plt.contourf(self.x, self.y, self.d)
		self.CS = self.plt.contourf(self.x, self.y, self.u)
		# self.CS = self.plt.contourf(self.x, self.y, self.v, np.linspace(-3,3,21))
		# self.CS = self.plt.plot(self.x[:,24], self.v[:,24])
    
	def plotConvergence(self):
		self.plt.plot(np.arange(1,len(self.residue),1), np.array(self.residue[1:])/self.residue[0])
		self.plt.yscale("log")
		self.plt.show()

	def plotVelocityVector(self, trim=lambda x: x[:24,24:72:2]):
		self.plt.quiver(*map(trim, [self.x, self.y, self.u, self.v]))
		self.plt.axis("equal")
		self.plotMesh()

	def plotStreamlines(self, trim=lambda x: x[:12,24:48]):
		self.plt.streamplot(*map(trim, [self.x, self.y, self.u, self.v]))
		self.plt.axis("equal")
		self.plotMesh()

	def animatefn(self, *args):
		self.marchOneStep()
		while self.iterations%self.plotInterval!=0:
			self.marchOneStep()
		self.plotField()
		self.log()
		self.saveData()
		self.saveAll()

	def solve(self, *args, **kwargs):
		self.animate = kwargs["animate"]
		if self.animate:
			self.animation = animation.FuncAnimation(self.fig, self.animatefn, interval=60)
			self.plt.hold(False)
			self.plt.colorbar(self.CS)
			# self.plt.show()
		else:
			while self.marchOneStep():
				if self.iterations%self.plotInterval == 0:
					# if self.animate:
					# 	self.plotField() 
					self.log()
					self.saveData()
					self.saveAll()
			self.plotConvergence()
			plt.show()
		self.log()

def parseCmdArgs():
	parser = OptionParser()
	
	parser.add_option("-r", "--resume", action="store_true", dest="resume", help="resume iteration")
	parser.add_option("-a", "--animate", action="store_true", dest="animate", help="animate the convergence ( really slow )")
	parser.add_option("-t", "--tolerance", action="store", dest="tolerance", default=1e-4, help="error tolerance level")
	parser.add_option("-f", "--file", action="store", dest="meshfile", default="bumpgrid.txt", help="filename containing mesh data")
	parser.add_option("-s", "--savefile", action="store", dest="savefile", default="data_dump.dat", help="file to save and resume intermediate result")
	parser.add_option("-i", "--logInterval", action="store", dest="logInterval", default=1000, help="no. of iterations before saving data and logging output")
	return parser.parse_args()

if __name__ == '__main__':
	options, args = parseCmdArgs()
	solver = Solver(
		meshfile=options.meshfile,
		resume=options.resume,
		savefile=options.savefile,
		plotInterval=int(options.logInterval),
		tolerance=float(options.tolerance)
		)
	solver.solve(
		animate=options.animate
		)
