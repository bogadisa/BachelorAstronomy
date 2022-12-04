import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

idata = np.loadtxt("IRAS13120_spectrum.txt")





window = np.array([-250, 250])   

def find_parameters():
    peak = np.max(idata[:, 1])
    #i_b = np.argmin(idata, axis=2)
     #wavelength_func(i_b)

    
    d = 0
    a = peak - d
    b = 81.644
    #lmbda_ = np.zeros((np.shape(b)[0], np.shape(b)[1], len(lmbda))) + lmbda
    # c = np.std(lmbda_ - b.reshape(550, 750, 1), axis=2)
    # c = FWHM(idata[:,0], idata[:, 1])
    p0 = [0, 1]
    errfunc = lambda p, x, y: g(x, p) - y # Distance to the target function
    p1, success = opt.leastsq(errfunc, p0[:], args=(idata[:, 0], idata[:, 1]))

    fit_mu, fit_stdev = p1

    FWHM = 334
    c = FWHM/(2*np.sqrt(2*np.log(2)))
    #FWHM = 2*np.sqrt(2*np.log(2))*fit_stdev
    #c = 223.103 #fit_stdev
    return a, b, c, d



n = len(idata)

a_ = 200

def g(x, p, manual=False):
    if manual:
        a, b, c = p
        print(a, b, c)
    else:
        c = p[1]
        b = p[0]
        a = 1/(c*np.sqrt(2*np.pi))
    return a*np.exp(-(x - b)**2/(2*c**2))

vmin, vmax = np.min(idata[:, 0]), np.max(idata[:, 0])
v = np.linspace(vmin, vmax, n)
a_, b, c, d = find_parameters()
print(a_, b, c, d)

#print(np.shape(a), np.shape(b), np.shape(c), np.shape(d))
T = g(v, [a_, b, c], manual=True)

plt.plot(idata[:, 0], idata[:, 1])
plt.plot(v, T)
plt.show()

Gamma_230Ghz = 37.5

plt.plot(idata[:, 0], Gamma_230Ghz*idata[:, 1])
plt.plot(v, Gamma_230Ghz*T)
plt.show()