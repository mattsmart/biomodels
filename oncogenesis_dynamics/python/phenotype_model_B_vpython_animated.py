from visual import *

N = 100.0
a=1.0
b=-0.2
c=0.5
alpha_plus = 10.0
alpha_minus = 11.0
mu = 0.2

scene.center = vector(33.3,33.3,33.3)
scene.autoscale=0
scene.range=(30,30,30)
lorenz = curve(color = color.green, radius=0.2 )
scene.background=color.white

### Draw grid
for x in arange(0,51,10):
    #curve( pos = [ (x,0,-25), (x,0,25) ], color = color.black, radius = 0.1 )
    box(pos=(x,0,0), axis=(0,0,50), height=0.4, width=0.4,color=color.black)
for z in arange(-25,26,10):
    #curve( pos = [ (0,0,z), (50,0,z) ], color = color.black, radius = 0.1 )
    box(pos=(25,0,z), axis=(50,0,0), height=0.4, width=0.4,color=color.black )

pointer = arrow(pos=(N,0,0), axis=(5,0,0), shaftwidth=3)
pointer = arrow(pos=(0,N,0), axis=(5,0,0), shaftwidth=3)
pointer = arrow(pos=(0,0,N), axis=(5,0,0), shaftwidth=3)
pyramid(pos=(0,0,0), axis=(0,0,1), size=(N/2,N/2,N), up=(0,0,1))
"""
mybox = box(pos=vector(x0,y0,z0), 
            axis=vector(a,b,c), length=L,
            height=H, width=W, up=vector(q,r,s))
#tr = paths.triangle(length=5)
mybox = box(pos=(N/3,N/3,N/3),
            axis=(0, -N/2, -N/2),
            length=N, height=N, width=0.1,
            up=vector(N/3,N/3,N/3)) 
"""

#dt = 0.0001
dt = 0.005
r = vector(90.0, 10.0, 0.0)  #initial cond make sure sum to N=100
#r = vector(30, 1, 7)

for t in arange(0,100,dt):
    fbar = (a*r.x + b*r.y + c*r.z) / N
    v = vector(-r.x*alpha_plus + r.y*alpha_minus        + (a - fbar)*r.x,
                r.x*alpha_plus - r.y*(alpha_minus + mu) + (b - fbar)*r.y,
                                 r.y*mu                 + (c - fbar)*r.z)
    r = r + v*dt
##  # Draw lines colored by speed
    cc = clip( [mag(v) * 0.005], 0, 1 )[0]
    lorenz.append( pos=r, color=(cc,0, 1-cc) )
    rate( 500 )
    print r, t

print 'done'
