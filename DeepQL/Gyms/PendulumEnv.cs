using DeepQL.Environments;
using DeepQL.Misc;
using DeepQL.Spaces;
using Neuro.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL.Gyms
{
    public class PendulumEnv : Env
    {
        public PendulumEnv()
            : base(new Box(-max_torque, max_torque, new Shape(1)),
                   new Box(new double[] { -1, -1 },
                           new double[] { 1, 1 }, new Shape(2)))
        {
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(500, 500);
                Viewer.SetBounds(-2.2, 2.2, -2.2, 2.2);
                var rod = Rendering.MakeCapsule(1, .2);
                rod.SetColor(.8, .3, .3);
                pole_transform = new Rendering.Transform();
                rod.AddAttr(pole_transform);
                Viewer.AddGeom(rod);
                var axle = Rendering.MakeCircle(.05);
                axle.SetColor(0, 0, 0);
                Viewer.AddGeom(axle);
                img = new Rendering.Image("assets/clockwise.png", 1.0, 1.0);
                imgtrans = new Rendering.Transform();
                img.AddAttr(imgtrans);
            }

            //Viewer.AddOneTime(self.img);
            pole_transform.SetRotation(State[0] + Math.PI / 2);
            if (last_u != double.NaN)
                imgtrans.SetScale(-last_u / 2, Math.Abs(last_u) / 2 );

            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(ObservationSpace.Shape);
            State.FillWithRand(-1, -1, 1);
            last_u = double.NaN;
            return State.Clone();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            double th = State[0], thdot = State[1]; // th := theta

            double g = 10.0;
            double m = 1.0;
            double l = 1.0;

            double u = action[0];
            u = Neuro.Tools.Clip(u, -max_torque, max_torque);
            last_u = u; // for rendering
            double costs = Math.Pow(AngleNormalize(th), 2) + 0.1 * Math.Pow(thdot, 2) + .001 * Math.Pow(u, 2);

            double newthdot = thdot + (-3 * g / (2 * l) * Math.Sin(th + Math.PI) + 3.0 / (m * Math.Pow(l, 2)) * u) * dt;
            double newth = th + newthdot * dt;
            newthdot = Neuro.Tools.Clip(newthdot, -max_speed, max_speed); //pylint: disable=E1111

            State = new Tensor(new[] { newth, newthdot }, ObservationSpace.Shape);
            observation = State.Clone();
            reward = -costs;

            return false;
        }

        private double AngleNormalize(double x)
        {
            return ((x + Math.PI) % (2 * Math.PI)) - Math.PI;
        }

        private Rendering.Viewer Viewer;
        private Rendering.Image img;
        private Rendering.Transform imgtrans;
        private Rendering.Transform pole_transform;
        private double last_u = double.NaN;

        private const double max_speed = 8;
        private const double max_torque = 2.0;
        private const double dt = .05;
    }
}
