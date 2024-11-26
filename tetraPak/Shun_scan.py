import shlex
import numpy as np
from sardana.macroserver.macro import macro, Type
from time import sleep
from time import time


@macro()
def Shun_long(self):
	for i in [40, 50, 60, 65, 70, 75, 80, 85, 90]:
		self.info(i)
		if i == 40:
			self.info(f'wait for 25 mins')
			for n in range(1500):
				sleep(1)
		self.execMacro(shlex.split('meshct_maxiv swaxs_x -1.25 1.25 50 sc_swaxs_yh -1.5 1.5 60 0.0498 True 0.002'))
		if i != 90:		
			self.info(f' sleeping for one hour')
			for n in range(3390):
				sleep(1)
		else:
			self.info(f' mission is completed')


@macro()
def Pia_long(self):
	for i in [25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190, 210, 270, 330, 390, 410]:
		self.info(i)
		if i == 25:
			self.info(f'wait for 15 mins')
			for n in range(600):
				sleep(1)
		#self.info(f' Scanning test')
		self.execMacro(shlex.split('meshct_maxiv swaxs_x -1.25 1.25 50 sc_swaxs_yh -1.5 1.5 60 0.0498 True 0.002'))
		if i <= 210:		
			self.info(f' sleeping for quarter of an hour')
			for n in range(690):
			#for n in range(1):		
				sleep(1)
		if i > 210:		
			self.info(f' Every hour')
			for n in range(3390):
			#for n in range(1):		
				sleep(1)
		if i > 270:		
			self.info(f' Every hour and a little bit')
			for n in range(1800):
			#for n in range(1):		
				sleep(1)
		if i != 410:		
			self.info(f' Next')					
		else:
			self.info(f' mission is completed')

''' 
@macro()
def shun_large_scan(self):
    #self.execMacro(shlex.split('umv swaxs_x -5'))
    for ii in range(1):
        for ii in range(2):
            self.execMacro(shlex.split('dscan swaxs_x -4 4 160 1 0.0015'))
            #self.execMacro(shlex.split('dscan swaxs_y -5 5 50 0.5 0.0015'))
            if ii <1:
                self.execMacro(shlex.split('umvr swaxs_x 20'))
            else:
                pass
        #self.execMacro(shlex.split('umvr swaxs_y -20'))
        #self.execMacro(shlex.split('umv swaxs_x -5'))
            #self.info(f' sleeping...')

'''
'''
@macro()
def shun_quick_scan(self):
   
    for ii in range(200):
        t0 = time.time()
        self.execMacro(shlex.split('timescan 0 1 0.0015'))
        now = time.time()
        if now-t0 < ??:
           self.warning('Elapsed time shorter than ??')
        while now-t0 < ??:
           sleep(0.1)
           now =time.time()
   
        #self.execMacro(shlex.split('dscan swaxs_y -5 5 50 0.5 0.0015'))
        #if ii <1:
        #    self.execMacro(shlex.split('umvr swaxs_x 20'))
        #else:
        #    pass
        #self.execMacro(shlex.split('umvr swaxs_y -20'))
        #self.execMacro(shlex.split('umv swaxs_x -5'))
            #self.info(f' sleeping...')

@macro()
def shun_hygro_scan(self):
    beamsize = 0.1
    Y = np.linspace(-3,3,int(6/beamsize)+1)
    #Y = np.linspace(-3, 3, 3)
    #X_temp = np.linspace(-3, 2, 6)
    #X_end = X_temp[-2]
    X = np.linspace(-3, 2, 6)
    #self.execMacro(shlex.split('umv swaxs_x -3 swaxs_y -3'))
    for n in range(10):
        for row in Y:
            self.execMacro(shlex.split('umv swaxs_y {}'.format(row)))
            for col in X:
                t0 = time()
                self.execMacro(shlex.split('umv swaxs_x {}'.format(col)))
                self.info('{} {}'.format(row, col))
                self.execMacro(shlex.split('timescan 0 1 0.015'))
                now = time()
                while now -t0 < 6.5:
                    sleep(0.1)
                    now = time()
                    self.info('timedifference is {}'.format(now-t0))
            #self.execMacro(shlex.split('umv swaxs_y {}'.format(row)))
            X = np.flip(X)
        X += beamsize
        Y = np.flip(Y)
    
    #for ii in range(200):
        #self.execMacro(shlex.split('timescan 0 1 0.0015'))
        #sleep(1)
        #self.execMacro(shlex.split('dscan swaxs_y -5 5 50 0.5 0.0015'))
        #if ii <1:
        #    self.execMacro(shlex.split('umvr swaxs_x 20'))
        #else:
        #    pass
        #self.execMacro(shlex.split('umvr swaxs_y -20'))
        #self.execMacro(shlex.split('umv swaxs_x -5'))
            #self.info(f' sleeping...')


@macro()
def shun_time_scan(self):

    self.execMacro(shlex.split('umv swaxs_x 0'))
    for n in range(1, 5):
        t0 = time()
        #self.execMacro(shlex.split('umv swaxs_x {}'.format(n/10)))
        self.info('{} {}'.format(1,2))
        self.execMacro(shlex.split('timescan 1 1 0.0015'))
        now = time()
        #dt = now - t0
        while now -t0 < 6.5:
            sleep(0.1)
            now = time()
        self.info('timedifference is {}'.format(now-t0))


@macro()
def shun_linkam_scan(self):

    num =250
    #timeinterval = 6
    
    for n in range(num):
        #t0 = time()
        
        #self.info('{} {}'.format(1,2))
        #self.execMacro(shlex.split('timescan 1 0.1 0.1'))
        self.execMacro(shlex.split('timescan 1 0.1'))
        #self.execMacro(shlex.split('dscan dummy_motor04 0 1 1 0.1'))
        sleep(24)
        #now = time()
        #dt = now - t0
        #while dt < timeinterval:
        #    sleep(0.1)
        #    now = time()
        #self.info('timedifference is {}'.format(now-t0))
'''
