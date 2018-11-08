using System;
using System.Collections.Generic;
using System.Linq;
using DeepQL.Environments;
using DeepQL.Misc;
using DeepQL.Spaces;
using Neuro.Tensors;

namespace DeepQL.Gyms
{
    public class AcrobotEnv : Env
    {
        /**
        Acrobot is a 2-link pendulum with only the second joint actuated
        Initially, both links point downwards. The goal is to swing the
        end-effector at a height at least the length of one link above the base.
        Both links can swing freely and can pass by each other, i.e., they don't
        collide when they have the same angle.
        **STATE:**
        The state consists of the sin() and cos() of the two rotational joint
        angles and the joint angular velocities :
        [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
        For the first link, an angle of 0 corresponds to the link pointing downwards.
        The angle of the second link is relative to the angle of the first link.
        An angle of 0 corresponds to having the same angle between the two links.
        A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
        **ACTIONS:**
        The action is either applying +1, 0 or -1 torque on the joint between
        the two pendulum links.
        .. note::
            The dynamics equations were missing some terms in the NIPS paper which
            are present in the book. R. Sutton confirmed in personal correspondence
            that the experimental results shown in the paper and the book were
            generated with the equations shown in the book.
            However, there is the option to run the domain with the paper equations
            by setting book_or_nips = 'nips'
        **REFERENCE:**
        .. seealso::
            R. Sutton: Generalization in Reinforcement Learning:
            Successful Examples Using Sparse Coarse Coding (NIPS 1996)
        .. seealso::
            R. Sutton and A. G. Barto:
            Reinforcement learning: An introduction.
            Cambridge: MIT press, 1998.
        .. warning::
            This version of the domain uses the Runge-Kutta method for integrating
            the system dynamics and is more realistic, but also considerably harder
            than the original version which employs Euler integration,
            see the AcrobotLegacy class.
        **/
        public AcrobotEnv()
            : base(new Discrete(3), 
                   new Box(new [] { -1, -1, -1, -1, -MAX_VEL_1, -MAX_VEL_2 },
                           new [] { 1, 1, 1, 1, MAX_VEL_1, MAX_VEL_2 }, new Shape(6)))
        {
            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            var s = State;

            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(500, 500);
                var bound = LINK_LENGTH_1 + LINK_LENGTH_2 + 0.2; // 2.2 for default
                Viewer.SetBounds(-bound, bound, -bound, bound);
            }

            if (s == null)
                return null;

            var p1 = new [] { LINK_LENGTH_1 * Math.Sin(s[0]), - LINK_LENGTH_1 * Math.Cos(s[0]) };
            //var p2 = new [] { p1[0] - LINK_LENGTH_2 * Math.Cos(s[0] + s[1]), p1[1] + LINK_LENGTH_2 * Math.Sin(s[0] + s[1]) };

            var xys = new List<double[]>{ new double[] {0, 0}, p1 };
            var thetas = new [] { s[0]-Math.PI/2, s[0]+s[1]- Math.PI / 2 };
            var linkLengths = new[] {LINK_LENGTH_1, LINK_LENGTH_2};

            Viewer.DrawLine(new []{-2.2, 1}, new []{2.2, 1});

            for (int i = 0; i < linkLengths.Length; ++i)
            {
                double x = xys[i][0], y = xys[i][1], th = thetas[i], llen = linkLengths[i];
                double l = 0, r = llen, t = .1,b = -.1;
                var jTransform = new Rendering.Transform(new []{x, y}, th);
                var link = Viewer.DrawPolygon(new List<double[]> { new[] { l, b }, new[] { l, t }, new[] { r, t }, new[] { r, b } });
                link.AddAttr(jTransform);
                link.SetColor(0, .8, .8);
                var circ = Viewer.DrawCircle(.1);
                circ.SetColor(.8, .8, 0);
                circ.AddAttr(jTransform);
            }

            Viewer.Render();
            return null;
        }

        protected override Tensor GetObservation()
        {
            var s = State;
            return new Tensor(new []{ Math.Cos(s[0]), Math.Sin(s[0]), Math.Cos(s[1]), Math.Sin(s[1]), s[2], s[3]}, ObservationSpace.Shape);
        }

        public override Tensor Reset()
        {
            State = new Tensor(new Shape(4));
            State.FillWithRand(-1, -0.1, 0.1);
            return GetObservation();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            var s = State;
            var a = (int)action[0];
            var torque = AVAIL_TORQUE[a];

            // Add noise to the force action
            if (TORQUE_NOISE_MAX > 0)
                torque += Rng.NextDouble(-TORQUE_NOISE_MAX, TORQUE_NOISE_MAX);

            // Now, augment the state with our force action so it can be passed to _dsdt
            double[] sAugmented = s.GetValues().Concat(new []{torque}).ToArray();

            var nsFull = Rk4(Dsdt, sAugmented, new [] { 0, DT });
            // only care about final timestep of integration returned by integrator
            //ns = ns[-1];
            //ns = ns[:4]  // omit action
            var ns = new double[4];
            var lastRow = nsFull.GetLength(0) - 1;
            for (int n = 0; n < 4; ++n)
                ns[n] = nsFull[lastRow, n];

            // ODEINT IS TOO SLOW!
            // ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
            // self.s_continuous = ns_continuous[-1] // We only care about the state
            // at the ''final timestep'', self.dt

            ns[0] = Wrap(ns[0], -Math.PI, Math.PI);
            ns[1] = Wrap(ns[1], -Math.PI, Math.PI);
            ns[2] = Bound(ns[2], -MAX_VEL_1, MAX_VEL_1);
            ns[3] = Bound(ns[3], -MAX_VEL_2, MAX_VEL_2);
            State = new Tensor(ns, State.Shape);
            bool terminal = Terminal();
            reward = terminal ? 0 : -1.0;
            observation = GetObservation();
            return terminal;
        }

        public override void Dispose()
        {
            Viewer.Dispose();
            Viewer = null;
            base.Dispose();
        }

        private bool Terminal()
        {
            var s = State;
            return -Math.Cos(s[0]) - Math.Cos(s[1] + s[0]) > 1.0;
        }

        private double[] Dsdt(double[] s_augmented, double t)
        {
            var m1 = LINK_MASS_1;
            var m2 = LINK_MASS_2;
            var l1 = LINK_LENGTH_1;
            var lc1 = LINK_COM_POS_1;
            var lc2 = LINK_COM_POS_2;
            var I1 = LINK_MOI;
            var I2 = LINK_MOI;
            var g = 9.8;
            var a = s_augmented.Last();
            var s = s_augmented.Take(s_augmented.Length - 1).ToArray();
            var theta1 = s[0];
            var theta2 = s[1];
            var dtheta1 = s[2];
            var dtheta2 = s[3];
            var d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * Math.Cos(theta2)) + I1 + I2;
            var d2 = m2 * (lc2 * lc2 + l1 * lc2 * Math.Cos(theta2)) + I2;
            var phi2 = m2 * lc2 * g * Math.Cos(theta1 + theta2 - Math.PI / 2.0);
            var phi1 = -m2 * l1 * lc2 * dtheta2 * dtheta2 * Math.Sin(theta2) -
                       2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * Math.Sin(theta2) +
                       (m1 * lc1 + m2 * l1) * g * Math.Cos(theta1 - Math.PI / 2) + phi2;

            var ddtheta2 = 0.0;

            if (BookOrNips == "nips")
            {
                // the following line is consistent with the description in the paper
                ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1);
            }
            else
            {
                // the following line is consistent with the java implementation and the book
                ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 * dtheta1 * Math.Sin(theta2) - phi2) / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1);
            }

            var ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
            return new[] {dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0};
        }

        private double Wrap(double x, double m, double M)
        {
            /**
            :param x: a scalar
            :param m: minimum possible value in range
            :param M: maximum possible value in range
            Wraps x so m <= x <= M; but unlike bound() which truncates, wrap() wraps x around the coordinate system defined by m,M.
            For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
            **/
            var diff = M - m;
            while (x > M)
                x = x - diff;
            while (x < m)
                x = x + diff;
            return x;
        }

        private double Bound(double x, double m, double M)
        {
            // bound x between min (m) and Max (M)
            return Math.Min(Math.Max(x, m), M);
        }

        private double Bound(double x, double[] m)
        {
            // bound x between min (m) and Max (M)
            return Math.Min(Math.Max(x, m[0]), m[1]);
        }

        private delegate double[] Derivs(double[] s_augmented, double t);

        private double[,] Rk4(Derivs derivs, double[] y0, double[] t)
        {
            /**
            Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
            This is a toy implementation which may be useful if you find
            yourself stranded on a system w/o scipy.  Otherwise use
            :func:`scipy.integrate`.
            *y0*
                initial state vector
            *t*
                sample times
            *derivs*
                returns the derivative of the system and has the
                signature ``dy = derivs(yi, ti)``
            Example 1 ::
                //// 2D system
                def derivs6(x,t):
                    d1 =  x[0] + 2*x[1]
                    d2 =  -3*x[0] + 4*x[1]
                    return (d1, d2)
                dt = 0.0005
                t = arange(0.0, 2.0, dt)
                y0 = (1,2)
                yout = rk4(derivs6, y0, t)
            Example 2::
                //// 1D system
                alpha = 2
                def derivs(x,t):
                    return -alpha*x + exp(-t)
                y0 = 1
                yout = rk4(derivs, y0, t)
            If you have access to scipy, you should probably be using the
            scipy.integrate tools rather than this function.
            **/

            var ny = y0.Length;
            var yOut = new double[t.Length, ny];

            for (int n = 0; n < ny; ++n)
                yOut[0,n] = y0[n];

            for (int i = 0; i < t.Length - 1; ++i)
            {
                var thisT = t[i];
                var dt = t[i + 1] - thisT;
                var dt2 = dt / 2.0;

                for (int n = 0; n < ny; ++n)
                    y0[n] = yOut[i, n];

                var k1 = derivs(y0, thisT);
                var k2 = derivs(y0.Zip(k1, (a,b) => a + dt2 * b).ToArray() , thisT + dt2);
                var k3 = derivs(y0.Zip(k2, (a, b) => a + dt2 * b).ToArray(), thisT + dt2);
                var k4 = derivs(y0.Zip(k3, (a, b) => a + dt * b).ToArray(), thisT + dt);

                for (int n = 0; n < ny; ++n)
                    yOut[i + 1, n] = y0[n] + dt / 6.0 * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]);
            }

            return yOut;
        }

        private const double DT = .2;

        private const double LINK_LENGTH_1 = 1.0;  // [m]
        private const double LINK_LENGTH_2 = 1.0;  // [m]
        private const double LINK_MASS_1 = 1.0;  //: [kg] mass of link 1
        private const double LINK_MASS_2 = 1.0;  //: [kg] mass of link 2
        private const double LINK_COM_POS_1 = 0.5;  //: [m] position of the center of mass of link 1
        private const double LINK_COM_POS_2 = 0.5;  //: [m] position of the center of mass of link 2
        private const double LINK_MOI = 1.0;  //: moments of inertia for both links

        private const double MAX_VEL_1 = 4 * Math.PI;
        private const double MAX_VEL_2 = 9 * Math.PI;

        private readonly double[] AVAIL_TORQUE = { -1.0, 0.0, 1.0 };

        private double TORQUE_NOISE_MAX = 0.0;

        //use dynamics equations from the nips paper or the book
        private string BookOrNips = "book";
        
        private Rendering.Viewer Viewer;
    }
}
