#-----------------------------------------------------------------------------#
#   Project 4                                                                 #
#-----------------------------------------------------------------------------#
#   Code used to simulate the dynamic of a quantum system subject to          #
#   a harmonic potential. The numeric simulation provided is a second order   #
#   Runge-Kutta simulation.                                                   #
#   
#                                                                             #
#   Course: Computational Methods in Physics (4300331)                        #
#   Professor: Luis Gregorio Dias da Silva                                    #
#   P.A's: Joao Victor Ferreira Alves and Lauro Barreto Braz                  #
#                                                                             #
#   Author: Rodrigo Pereira Silva. NUSP: 11277128                             #
#   Contact: rodrigopereirasilva@usp.br                                       #
#-----------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib as mpl

def psi0(omega, x, n):
    '''
    This function calculates the initial states according to Hermite 
    polynomials

    Parameters
    ----------
    omega : Float
        Frequency of oscillation.
    x : Array
        Array of positions along x-axis.
    n : Int
        Mode of vibration.

    Returns
    -------
    Real-valued Array
        Returns an array with the initial wave function.
    '''
    hermite_pol = special.hermite(n, monic=False)
    return (1/np.sqrt(2**n*special.factorial(n)))*\
            (omega/np.pi)**(1/4)*np.exp(-omega*x**2/2)*hermite_pol(np.sqrt(omega)*x)


def SchrodingerRK2(tf, dt, r, L, omega, n, k0):
    '''
    

    Parameters
    ----------
    tf : Float
        Duration of the Runge-Kutta simulation.
    dt : Float
        Time-step.
    r : Float
        Parameter used to calculate space discretization.
    L : Float
        Region considered for the simulation.
    omega : Float
        Frequency of oscillation.
    n : Int
        Mode of vibration.
    k0 : Float
        Initial momentum.

    Returns
    -------
    Psi : Complex-valued Array
        Wave-function.
    N : Real-valued Array
        Normalization vector for all times considered.
    E : Complex-Valued Array
        Energy vector for all times considered.

    '''
    dx = np.sqrt(dt/(2*r))
    x = -np.arange(-L,L,dx)
    t = np.arange(0,tf+dt,dt)
    tsize = len(t) ; xsize = len(x) 

    R = np.zeros( (xsize,tsize) ) #Real part
    I = np.zeros( (xsize,tsize) ) #Complex part
    Psi = np.zeros( (xsize,tsize), dtype=np.complex64 )
    V = (1/2)*(omega*x)**2
    
    k1I = np.zeros( xsize ) # k1 for complex part 
    k1R = np.zeros( xsize ) # k1 for real part
    k2I = np.zeros( xsize ) # k2 for complex part
    k2R = np.zeros( xsize ) # k2 for real part

    #half-step values
    R12 = np.zeros(xsize)
    I12 = np.zeros(xsize)
    
    # real and imaginary parts of the Hamiltonian
    HR = np.zeros( (xsize, tsize) )
    HI = np.zeros( (xsize, tsize) )

    E = np.zeros(tsize, dtype=np.complex64)

    N = np.zeros(tsize) # normalization integral
    
    #definindo as condicoes iniciais da funcao de onda
    Psi[:,0] = psi0(omega, x, n)*np.exp(1j*k0*x) #initial condition
    R[:,0] = Psi[:,0].real
    I[:,0] = Psi[:,0].imag
    
    #time loop for Runge-Kutta
    for n in range(tsize-1):
        #impsing the boundary conditions
        I[0,:]  = 0
        I[-1,:] = 0
        R[0,:]  = 0
        R[-1,:] = 0
        
        #using array slicing it is possible to avoid space loops
        #shoutout to my friend Rodrigo Da Motta, who suggested this clever
        #solution to improve algorithm's efficiency
        
        #calculating k1R's
        k1R[1:-1] = -r*(I[2:,n] - 2*I[1:-1, n] + I[:-2, n]) + V[1:-1]*I[1:-1,n]*dt
        k1R[0]    = -r*(I[1,n] - 2*I[0,n] + I[-1,n]) + V[0]*I[0,n]*dt
        k1R[-1]   = -r*(I[0,n] - 2*I[-1,n] + I[-2,n]) + V[-1]*I[-1,n]*dt
        
        #calculating k1I's
        k1I[1:-1] = +r*(R[2:,n] - 2*R[1:-1,n] + R[:-2,n] ) - V[1:-1]*R[1:-1,n]*dt
        k1I[0]    = +r*(R[1,n] - 2*R[0,n] + R[-1,n]) - V[0]*R[0,n]*dt
        k1I[-1]   = +r*(R[0,n] - 2*R[-1,n] + R[-2,n]) - V[-1]*R[-1,n]*dt
        
        #calculating half-step values
        R12       = R[:,n] + k1R/2
        I12       = I[:,n] + k1I/2
        
        #calculating k2R's
        k2R[1:-1] = -r*(I12[2:] - 2*I12[1:-1] + I12[:-2]) + V[1:-1]*I12[1:-1]*dt
        k2R[0]    = -r*(I12[1] - 2*I12[0] + I12[-1]) + V[0]*I12[0]*dt
        k2R[-1]   = -r*(I12[0] - 2*I12[-1] + I12[-2]) + V[-1]*I12[-1]*dt
        
        #calculating k2I's
        k2I[1:-1] = +r*(R12[2:] - 2*R12[1:-1] + R12[:-2]) - V[1:-1]*R12[1:-1]*dt
        k2I[0]    = +r*(R12[1] - 2*R12[0] + R12[-1]) - V[0]*R12[0]*dt
        k2I[-1]   = +r*(R12[0] - 2*R12[-1] + R12[-2]) - V[-1]*R12[-1]*dt 
        
        #time evolution of the real and imaginary parts
        R[:,n+1] = R[:,n] + k2R
        I[:,n+1] = I[:,n] + k2I
        R[:,-1]  = R[:,-2] + k2R
        I[:,-1]  = I[:,-2] + k2I
        
        #re-imposing boundary conditions
        I[0,:]   = 0
        I[-1,:]  = 0
        R[0,:]   = 0
        R[-1,:]  = 0

        #calculating HI
        HR[1:-1,n] = -(1/(2*dx**2))*( R[2:,n] - 2*R[1:-1,n] + R[:-2,n] ) + V[1:-1]*R[1:-1,n]
        HR[-1,n]   = -(1/(2*dx**2))*( R[0,n] - 2*R[-1,n] + R[-2,n] ) + V[-1]*R[-1,n]
        HR[0,n]    = -(1/(2*dx**2))*( R[1,n] - 2*R[0,n] + R[-1,n] ) + V[0]*R[0,n]
        
        #calculating HR
        HI[1:-1,n] = -(1/(2*dx**2))*( I[2:,n] - 2*I[1:-1,n] + I[:-2,n] ) + V[1:-1]*I[1:-1,n]
        HI[0,n]    = -(1/(2*dx**2))*( I[-1,n] - 2*I[0,n] + I[1,n] ) + V[0]*I[0,n]
        HI[-1,n]   = -(1/(2*dx**2))*( I[-2,n] - 2*I[-1,n] + I[0,n] ) + V[-1]*I[-1,n]

    #calculating HR at the last time step
    HR[1:-1,-1] = -(1/2)*((R[2:,-1] - 2*R[1:-1,-1] + R[:-2,-1])/(dx**2)) + V[1:-1]*R[1:-1,-1]
    HR[0,-1]    = -(1/2)*((R[1,n] - 2*R[0,-1] + R[-1,-1])/(dx**2)) + V[0]*R[0,-1]
    HR[-1,-1]   = -(1/2)*((R[0,n] - 2*R[-1,-1] + R[-2, -1])/(dx**2)) + V[-1]*R[-1,-1]
    
    #calculating HI at the last time step
    HI[1:-1,-1] = -(1/2)*((I[2:,-1] - 2*I[1:-1,-1] + I[:-2,-1])/(dx**2)) + V[1:-1]*I[1:-1,-1]
    HI[0,-1]    = -(1/2)*((I[1,-1] - 2*I[0,-1] + I[-1,-1])/(dx**2)) + V[0]*I[0,-1]
    HI[-1,-1]   = -(1/2)*((I[0,-1] - 2*I[-1,-1] + I[-2,-1])/(dx**2)) + V[-1]*I[-1,-1]
    
    #defining wave function from real and imaginary parts
    Psi = R + 1j*I
    
    #calculating normalization of the wave equation
    N = np.trapz(np.abs(Psi)**2, dx=dx, axis=0)
    
    #calculating energy from Hamiltonian real and imaginary parts
    E = np.trapz(np.conjugate(Psi)*(HR + 1j*HI), dx=dx, axis=0 ) / N
    
    return (Psi, N, E, V, x, t, xsize, tsize)

#Examples of code implementation:

#-----------------------------------------------------------------------------#
# PDF FUNCTION FOR THE FIRST EXCITED STATE                                    #
#-----------------------------------------------------------------------------#
Psi, N, E, V, x, t, xsize, tsize = SchrodingerRK2(tf=0.05, dt=1e-5, r=0.075, L=1, omega=50, n=1, k0=0)
fig1 = plt.figure(figsize=(6,5))
ax1 = fig1.add_axes([0,0,1,1])
ax1.plot(x, np.abs(Psi[:,0])**2)
ax1.set_xlabel('x')
ax1.set_ylabel(r'$|\psi(x,0)|^2$')

plt.show()



#-----------------------------------------------------------------------------#
# PDF OF FIRST MODES OF VIBRATION                                             #
#-----------------------------------------------------------------------------#

Psi0 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=0, k0=0)[0]
Psi1 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=1, k0=0)[0]
Psi2 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=2, k0=0)[0]
Psi3 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=3, k0=0)[0]
Psi4 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=4, k0=0)[0]
Psi5 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=5, k0=0)[0]
Psi6 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=6, k0=0)[0]
Psi7 = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=7, k0=0)[0]

x = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=0, k0=0)[-4]
V = SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=5, omega=1, n=0, k0=0)[3]

fig = plt.figure(figsize=(6,5))
ax = fig.add_axes([0,0,1,1])

ax.plot(x, 1*7.5 + np.abs(Psi7[:,10])**2, label=r'$P_7(x)$')
ax.plot(x, 1*6.5 + np.abs(Psi6[:,10])**2, label=r'$P_6(x)$')
ax.plot(x, 1*5.5 + np.abs(Psi5[:,10])**2, label=r'$P_5(x)$')
ax.plot(x, 1*4.5 + np.abs(Psi4[:,10])**2, label=r'$P_4(x)$')
ax.plot(x, 1*3.5 + np.abs(Psi3[:,10])**2, label=r'$P_3(x)$')
ax.plot(x, 1*2.5 + np.abs(Psi2[:,10])**2, label=r'$P_2(x)$')
ax.plot(x, 1*1.5 + np.abs(Psi1[:,10])**2, label=r'$P_1(x)$')
ax.plot(x, 1*0.5 + np.abs(Psi0[:,10])**2, label=r'$P_0(x)$')
ax.plot(x, V, color='black', linestyle='dashed', label=r'$V(x)$')

ax.set_xlim(-5,5)
ax.set_ylim(0,8.5)
ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
ax.set_yticklabels([r'$E_0$', r'$E_1$', r'$E_2$', r'$E_3$', r'$E_4$', r'$E_5$', r'$E_6$', r'$E_7$'], fontsize=26)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_xticklabels([-4, -2, 0, 2, 4], fontsize=26)
ax.set_xlabel(r'$x$', fontsize=30)
ax.set_ylabel(r'$|\psi_n(x,t)|^2 + \omega (n + 1/2)$', fontsize=30)
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

plt.show()



#-----------------------------------------------------------------------------#
# ENERGIES FOR THE FIRST EIGHT STATES                                         #
#-----------------------------------------------------------------------------#

Ergs = np.array([SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=1, omega=50, n=i, k0=0)[2][0] for i in range(8)]) #energies at t0
Ergs_final = np.array([SchrodingerRK2(tf=0.05, dt=0.00001, r=0.075, L=1, omega=50, n=i, k0=0)[2][-1] for i in range(8)]) #energies at ft for later comparison

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_axes([0,0,1,1])
n = np.arange(0,8,1)
ax1.plot(n, [50*0.5, 50*1.5, 50*2.5, 50*3.5, 50*4.5, 50*5.5, 50*6.5, 50*7.5], color='red', label='Theoretical')
ax1.scatter(n, Ergs.real, color='black', label='Numeric')
ax1.set_xlabel(r'$n$',fontsize=30)
ax1.set_ylabel(r'$E_n$', fontsize=30)
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax1.set_yticks([100, 200, 300])
ax1.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=24)
ax1.set_yticklabels(['100', '200', '300'], fontsize=24)
plt.legend()
plt.grid()
plt.show()

#energy difference of initial and final times
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_axes([0,0,1,1])
n = np.arange(0,8,1)
ax1.plot(n, (Ergs - Ergs_final).real, 'o-')
ax1.set_xlim(0,7)
ax1.set_xlabel(r'$n$', fontsize=30)
ax1.set_ylabel(r'$\Delta_t E \;\; (\times 10^{-7})$', fontsize=30)
ax1.set_xticks([0,1,2,3,4,5,6,7])
ax1.set_xticklabels([0,1,2,3,4,5,6,7], fontsize=24)
ax1.set_yticks([1e-7*0,1e-7*-1,1e-7*-2,1e-7*-3,1e-7*-4,1e-7*-5])
ax1.set_yticklabels([0,-1,-2,-3,-4,-5], fontsize=24)
ax1.grid()
plt.show()



#-----------------------------------------------------------------------------#
# CHECKING UNCERTAINTY PRINCIPLE                                              #
#-----------------------------------------------------------------------------#

def uncertainty(Psi, dt, r):
    dx = np.sqrt(dt/(2*r))
    diff_Psi = np.gradient(Psi[:,0], dx )  #first derivative
    diff2_Psi = np.gradient(diff_Psi, dx ) #second derivative

    mean_p = -1j*np.trapz(np.conjugate(Psi[:,0])*diff_Psi, dx=dx )
    mean_p2 = -np.trapz(np.conjugate(Psi[:,0])*diff2_Psi, dx=dx )
    mean_x = np.trapz(np.conjugate(Psi[:,0])*x*Psi[:,0], dx=dx )
    mean_x2 = np.trapz(np.conjugate(Psi[:,0])*(x**2)*Psi[:,0], dx=dx )

    sigma_x = np.sqrt( mean_x2 - mean_x**2 )
    sigma_p = np.sqrt( mean_p2 - mean_p**2 )

    return (sigma_x*sigma_p, mean_p, mean_p2, mean_x, mean_x2)

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_axes([0,0,1,0.9])

n = np.arange(0,8,1)
ax1.scatter(n, [uncertainty(Psi=Psi0, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi1, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi2, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi3, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi4, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi5, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi6, dt=0.00001, r=0.075)[0].real,
           uncertainty(Psi=Psi7, dt=0.00001, r=0.075)[0].real], color='black', label='Numeric')
ax1.plot(n, n+(1/2), color='red', label='Theoretical')
ax1.set_xlabel(r'$n$', fontsize=30)
ax1.set_ylabel(r'$\sigma_x \sigma_p$', fontsize=30)
ax1.set_xticks([0,1,2,3,4,5,6,7])
ax1.set_xticklabels(['0', '1', '2', '3', '4', '5', '6' ,'7'], fontsize=30)
ax1.set_yticks([0.5, 2.5, 4.5, 6.5, 8.5])
ax1.set_yticklabels(['0.5', '2.5', '4.5', '6.5', '8.5'], fontsize=30)
ax1.set_xlim(-0.1,7.1)
ax1.set_ylim(0.3, 7.6)
ax1.grid()
ax1.legend()

plt.show()


#-----------------------------------------------------------------------------#
# CLASSICAL AND QUANTUM OSCILLATOR                                            #
#-----------------------------------------------------------------------------#

Psi20 = SchrodingerRK2(tf=0.05, dt=0.000001, r=0.075, L=1, omega=50, n=20, k0=0)[0]
x = SchrodingerRK2(tf=0.05, dt=0.000001, r=0.075, L=1, omega=50, n=20, k0=0)[4]
A = np.sqrt((2/50)*(20 + 1/2))
fig = plt.figure(figsize=(6,5))
ax1 = fig.add_axes([0,0,1,1])

ax1.plot(x , np.abs(Psi20[:,0])**2, label='Quantum', color='black')
ax1.plot(x, (1/np.pi)*(1/(np.sqrt(A**2 - x**2))), linestyle='dashed', label='Classical', color='red')
ax1.set_xlabel(r'$x$', fontsize=30)
ax1.set_ylabel(r'$|\psi(x)|^2$', fontsize=30)
ax1.grid()
ax1.set_xlim(-1,1)
ax1.set_ylim(0,2)
ax1.set_yticks([0, 0.5, 1, 1.5, 2])
ax1.set_yticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=30)
ax1.set_xticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'], fontsize=30)
ax1.legend()

plt.show()



#-----------------------------------------------------------------------------#
# ANIMATION WITH EXPECTED <X>                                                 #
#-----------------------------------------------------------------------------#

#I'm not sure this will work properly, since I created these animations
#on jupyter notebook. But the scrip is as follows:

#definition of a series of parameters to be used in the animation function
Psi_ani, N_ani, E_ani, V_ani, x_ani, t_ani, xsize_ani, tsize_ani = SchrodingerRK2(tf=0.125, dt=0.00001, r=0.075, L=1, omega=50, n=0, k0=10)
tf = 0.125
dt = 1e-5
mpl.rcParams['animation.embed_limit'] = 2**128
fig = plt.figure(figsize=(10,6))
ax = fig.add_axes([0,0,1,1])
pico = np.max(np.abs(Psi_ani[:, 0])**2)
step = int(tsize/300)
time = np.arange(0, tf, dt)

dx = np.sqrt(1e-5/(2*0.075))
omega = 50

mean_x0 = np.trapz(np.conjugate(Psi_ani[:,0])*x_ani*Psi_ani[:,0], dx = dx )

diff_Psi = np.gradient(Psi_ani[:,0], dx )
mean_p0 = np.trapz(np.conjugate(Psi_ani[:,0])*(-1j)*diff_Psi, dx=dx)
X = mean_x0.real*np.cos(omega*t_ani) - (1/omega)*mean_p0.real*np.sin(omega*t_ani)


def animate(i):
    ax.clear()
    ax.clear()
    ax.plot(x_ani, np.abs(Psi_ani[:, step*i])**2, label=r'$| \psi (x, t) |^2$', color='black')
    ax.vlines(x=np.real(X[step*i]), ymin=0, ymax=np.max(np.abs(Psi_ani[:, step*i])**2), color='red', 
              ls='--', lw=1, label=r'$\langle \hat{x} \rangle$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$| \psi (x, t) |^2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.2, 4.2)
    ax.legend()
    
ani = FuncAnimation(fig, animate, frames=int(tsize/step), interval=33, repeat=False)
f = r'x_mean_animation_spydertest.gif'
writergif = PillowWriter(fps=30) 
ani.save(f, writer=writergif)

