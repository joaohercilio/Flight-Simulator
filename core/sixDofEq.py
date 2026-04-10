def navigationEquations(u, v, w, sphi, cphi, stet, ctet, spsi, cpsi):
	xdot = u*ctet*cpsi + v*( sphi*stet*cpsi - cphi*spsi ) + \
			w*( cphi*stet*cpsi + sphi*spsi )
	ydot = u*ctet*spsi + v*( sphi*stet*spsi + cphi*cpsi ) + \
			w*( cphi*stet*spsi-sphi*cpsi )
	zdot = -u*stet + v*sphi*ctet + w*cphi*ctet
	return xdot, ydot, zdot

def kinematicEquations(p, q, r, sphi, cphi, ctet, ttet):
	phidot = p + ( q*sphi + r*cphi )*ttet
	tetadot = q*cphi - r*sphi
	psidot = ( q*sphi + r*cphi )/ctet
	return phidot, tetadot, psidot

def translationalEquations(mass, g, X, Y, Z, u, v, w, p, q, r, sphi, cphi, stet, ctet):
	udot = X/mass - g*stet - q*w + r*v
	vdot = Y/mass + g*ctet*sphi - r*u + p*w
	wdot = Z/mass + g*ctet*cphi - p*v + q*u
	return udot, vdot, wdot

def rotationalEquations(Ix, Iy, Iz, Ixz, l, m, n, p, q, r):
	den = Ix*Iz - Ixz**2
	pdot = ( Ixz*( Ix-Iy+Iz )*p*q - ( Iz*(Iz-Iy) + Ixz**2 )*q*r + Iz*l + Ixz*n )/den
	qdot = ( ( Iz-Ix )*r*p - Ixz*(p**2 - r**2) + m )/Iy
	rdot = (-Ixz*( Ix-Iy+Iz )*q*r + ( Ix*(Ix-Iy) + Ixz**2 )*p*q + Ixz*l + Ix*n )/den
	return pdot, qdot, rdot
