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
    public class MountainCarEnv : Env
    {
        public MountainCarEnv()
            : base(new Discrete(3), new Box(new double[] { min_position, -max_speed },
                                            new double[] { max_position, max_speed }, new Shape(2)))
        {
            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            const int SCREEN_WIDTH = 600;
            const int SCREEN_HEIGHT = 400;

            double world_width = max_position - min_position;
            double scale = SCREEN_WIDTH / world_width;
            const int carwidth = 40;
            const int carheight = 20;

            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT);
                var xs = Neuro.Tools.LinSpace(min_position, max_position, 100);
                var ys = xs.Select(x => Height(x)).ToList();
                var xys = xs.Zip(ys, (x, y) => new[] { (x - min_position) * scale, y * scale }).ToList();

                Rendering.PolyLine track = Rendering.MakePolyLine(xys) as Rendering.PolyLine;
                track.SetLineWidth(4);
                Viewer.AddGeom(track);

                double clearance = 10;

                double l = -carwidth / 2, r = carwidth / 2, t = carheight, b = 0;
                var car = new Rendering.FilledPolygon(new List<double[]> { new[] { l, b }, new[] { l, t }, new[] { r, t }, new[] { r, b } });
                car.AddAttr(new Rendering.Transform(new[] { 0, clearance }));
                CarTrans = new Rendering.Transform();
                car.AddAttr(CarTrans);
                Viewer.AddGeom(car);
                var frontwheel = Rendering.MakeCircle(carheight / 2.5);
                frontwheel.SetColor(.5, .5, .5);
                frontwheel.AddAttr(new Rendering.Transform(new[] { carwidth / 4, clearance }));
                frontwheel.AddAttr(CarTrans);
                Viewer.AddGeom(frontwheel);
                var backwheel = Rendering.MakeCircle(carheight / 2.5);
                backwheel.AddAttr(new Rendering.Transform(new[] { -carwidth / 4, clearance }));
                backwheel.AddAttr(CarTrans);
                backwheel.SetColor(.5, .5, .5);
                Viewer.AddGeom(backwheel);
                double flagx = (goal_position - min_position) * scale;
                double flagy1 = Height(goal_position) * scale;
                double flagy2 = flagy1 + 50;
                var flagpole = new Rendering.Line(new[] { flagx, flagy1 }, new[] { flagx, flagy2 });
                Viewer.AddGeom(flagpole);
                var flag = new Rendering.FilledPolygon(new List<double[]>() { new[] { flagx, flagy2 }, new[] { flagx, flagy2 - 10 }, new[] { flagx + 25, flagy2 - 5 } });
                flag.SetColor(.8, .8, 0);
                Viewer.AddGeom(flag);
            }

            double pos = State[0];
            CarTrans.SetTranslation((pos - min_position) * scale, Height(pos) * scale);
            CarTrans.SetRotation(Math.Cos(3 * pos));

            Viewer.Render();

            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(new[] { Rng.NextDouble(-0.6, -0.4), 0.0}, ObservationSpace.Shape);
            return State.Clone();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            double position = State[0];
            double velocity = State[1];

            velocity += ((int)action[0] - 1) * 0.001 + Math.Cos(3 * position) * (-0.0025);
            velocity = Neuro.Tools.Clip(velocity, -max_speed, max_speed);
            position += velocity;
            position = Neuro.Tools.Clip(position, min_position, max_position);
            if (position <= min_position && velocity < 0)
                velocity = 0;

            bool done = position >= goal_position;
            reward = -1.0;

            State = new Tensor(new[] { position, velocity }, ObservationSpace.Shape);

            observation = State.Clone();
            return done;
        }

        private double Height(double xs)
        {
            return Math.Sin(3 * xs) * .45 + .55;
        }

        private Rendering.Viewer Viewer;
        private Rendering.Transform CarTrans;

        private const double min_position = -1.2;
        private const double max_position = 0.6;
        private const double max_speed = 0.07;
        private const double goal_position = 0.5;
    }
}
