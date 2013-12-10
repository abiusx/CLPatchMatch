'''
CLPatchMatch by AbiusX
This code runs the patch match algorithm on two images, using OpenCL and runs in realtime.

Performance:
50 iterations of this take less than a second on the development machine, where as the sequential
code requires 5+ seconds for each iteration.
'''
import pyopencl as cl
import numpy
import pylab
import matplotlib;
import skimage
import skimage.io
import skimage.transform
import datetime

from operator import itemgetter
import sys
import math
import random
import os


files=["bike_a.png","bike_b.png"];
def getTime():
    return datetime.datetime.now();

class CLPatchMatch:
    '''
    CLSeamCarving class,
    performs the seam carving algorithm on an image to reduce its size without scaling
    its main features, using OpenCL in realtime
    '''
    def __init__(self):
        '''
        Inits the opencl environment and parameters
        '''
        #profiling operation times
        self.times={k:datetime.timedelta() for k in ("execute",
                    "init","randomfill")};
        #show opencl compiler errors
        os.environ["PYOPENCL_COMPILER_OUTPUT"]="1";
        
        t=getTime();

        self.ctx = cl.create_some_context(False)
        self.queue = cl.CommandQueue(self.ctx)
        
        self.times["init"]+=getTime()-t;
        
    def loadProgram(self, filename):
        t=getTime();
        
        f = open(filename, 'r')
        fstr = "".join(f.readlines())

        #create the program
        self.program = cl.Program(self.ctx, fstr).build()
        self.times["init"]+=getTime()-t;

    def loadImages(self,files):
        '''
        load both images into self.img[0,1]
        '''
        t=getTime();
        self.img = [skimage.img_as_float(skimage.io.imread(files[i])) for i in (0,1)]
        self.loadProgram("patchmatch.c")
        self.times["init"]+=getTime()-t;

    def randomfill(self):
        t=getTime();
        mf= cl.mem_flags
        self.inputBuf=[cl.Buffer(self.ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.img[i]) for i in [0,1]];
        self.outputBuf=cl.Buffer(self.ctx,mf.WRITE_ONLY | mf.COPY_HOST_PTR ,hostbuf=self.nff)
        
        
        self.program.randomfill(self.queue, self.effectiveSize, None, 
                                  numpy.int32(self.patchSize[0]), #patchHeight
                                  numpy.int32(self.patchSize[1]), #patchWidth
                                  numpy.int32(self.size[0]), #height
                                  numpy.int32(self.size[1]), #width
                                   self.inputBuf[0],
                                   self.inputBuf[1],
                                   self.outputBuf)
        c = numpy.empty_like(self.nff)
        cl.enqueue_read_buffer(self.queue, self.outputBuf, c).wait()
        self.nff=numpy.copy(c);

        self.times["randomfill"]+=getTime()-t;

    def execute(self):
        '''
        execute an iteration of patchMatch
        '''
        t=getTime();
        mf= cl.mem_flags
        self.inputBuf=[cl.Buffer(self.ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.img[i]) for i in [0,1]];
        self.outputBuf=cl.Buffer(self.ctx,mf.READ_WRITE | mf.COPY_HOST_PTR ,hostbuf=self.nff)
        
        
        self.program.propagate(self.queue, self.effectiveSize, None, 
                                  numpy.int32(self.patchSize[0]), #patchHeight
                                  numpy.int32(self.patchSize[1]), #patchWidth
                                  numpy.int32(self.size[0]), #height
                                  numpy.int32(self.size[1]), #width
                                  numpy.int32(self.iteration),
                                   self.inputBuf[0],
                                   self.inputBuf[1],
                                   self.outputBuf)
        c = numpy.empty_like(self.nff)
        cl.enqueue_read_buffer(self.queue, self.outputBuf, c).wait()
        self.nff=numpy.copy(c);

        self.times["execute"]+=getTime()-t;

    def _drawRect(self,img,y,x,height,width,color=(1,0,0)):
        '''
        used for demo, showing which rectangles match
        '''
        for i in range(0,width+1):
            img[y][x+i]=color;
            img[y+height][x+i]=color;
        for i in range(0,height+1):
            img[y+i][x]=color
            img[y+i][x+width]=color;
        
    def show(self,nffs=True):
        '''
        shows times and images
        '''
        samples=5;
        for i in range(0,samples):
            color=(random.random(),random.random(),random.random())
            randomPoint=[(int)(random.random()*i) for i in self.effectiveSize];
            self._drawRect(self.img[0],randomPoint[0],randomPoint[1],self.patchSize[0],self.patchSize[1],color);
            self._drawRect(self.img[1],self.nff[randomPoint[0]][randomPoint[1]][0],self.nff[randomPoint[0]][randomPoint[1]][1],self.patchSize[0],self.patchSize[1],color);

        for i in self.times.keys():
            print i,":", (self.times[i].seconds*1000+self.times[i].microseconds/1000)/1000.0,"seconds"
        
        if nffs:
            for i in range(0,3):
                pylab.imshow(self.nff[:,:,i])
                pylab.show();
        f=pylab.figure()
        f.add_subplot(1,2,0);    
        pylab.imshow(self.img[0],cmap=matplotlib.cm.Greys_r);
        f.add_subplot(1,2,1);
        pylab.imshow(self.img[1],cmap=matplotlib.cm.Greys_r);
        pylab.title("Patch Match (by AbiusX)")
        pylab.show();
    
    def D(self,y1,x1,y2,x2):
        '''
        compute difference of two patches
        '''
        return ((self.img[0][y1:y1+self.patchSize[0],x1:x1+self.patchSize[1]]-
                self.img[1][y2:y2+self.patchSize[0],x2:x2+self.patchSize[1]])**2).sum();

    def match(self,files,patchSize=(7,7),iterations=20,Demo=False):
        '''
        run the patchMatch algorithm on the images, returning nff array
        '''
        self.loadImages(files);

        
        self.size=self.img[0].shape;
        self.patchSize=patchSize;
        self.effectiveSize=[self.size[i]-patchSize[i] for i in (0,1)];
        self.nff=numpy.ndarray((self.effectiveSize[0],self.effectiveSize[1],3));

        self.randomfill();

        for i in range(0,iterations):
            self.iteration=i+1;
            if (Demo):
                print "iteration",self.iteration
                print "mean block difference:", self.nff[:,:,2].mean();
            self.execute();
        
        if (Demo): self.show();
        return self.nff;


if __name__ == "__main__":
    patchmatch = CLPatchMatch()
    print "Please wait a few seconds...";
    patchmatch.match(files,Demo=True);
    print "Done."
