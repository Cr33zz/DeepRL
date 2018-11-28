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
            : base(new Discrete(3), 
                   new Box(new [] { min_position, -max_speed },
                           new [] { max_position, max_speed }, new Shape(2)))
        {
            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            const int SCREEN_WIDTH = 600;
            const int SCREEN_HEIGHT = 400;

            float world_width = max_position - min_position;
            float scale = SCREEN_WIDTH / world_width;
            const int carwidth = 40;
            const int carheight = 20;

            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT);
                var xs = Neuro.Tools.LinSpace(min_position, max_position, 100);
                var ys = xs.Select(x => Height(x)).ToList();
                var xys = xs.Zip(ys, (x, y) => new[] { (x - min_position) * scale, y * scale }).ToList();

                Rendering.Polyline track = Rendering.MakePolyLine(xys) as Rendering.Polyline;
                track.SetLineWidth(4);
                Viewer.AddGeom(track);

                float clearance = 10;

                float l = -carwidth / 2, r = carwidth / 2, t = carheight, b = 0;
                var car = new Rendering.FilledPolygon(new List<float[]> { new[] { l, b }, new[] { l, t }, new[] { r, t }, new[] { r, b } });
                car.AddAttr(new Rendering.Transform(new[] { 0, clearance }));
                CarTrans = new Rendering.Transform();
                car.AddAttr(CarTrans);
                Viewer.AddGeom(car);
                var frontwheel = Rendering.MakeCircle(carheight / 2.5f);
                frontwheel.SetColor(.5f, .5f, .5f);
                frontwheel.AddAttr(new Rendering.Transform(new[] { carwidth / 4, clearance }));
                frontwheel.AddAttr(CarTrans);
                Viewer.AddGeom(frontwheel);
                var backwheel = Rendering.MakeCircle(carheight / 2.5f);
                backwheel.AddAttr(new Rendering.Transform(new[] { -carwidth / 4, clearance }));
                backwheel.AddAttr(CarTrans);
                backwheel.SetColor(.5f, .5f, .5f);
                Viewer.AddGeom(backwheel);
                float flagx = (goal_position - min_position) * scale;
                float flagy1 = Height(goal_position) * scale;
                float flagy2 = flagy1 + 50;
                var flagpole = new Rendering.Line(new[] { flagx, flagy1 }, new[] { flagx, flagy2 });
                Viewer.AddGeom(flagpole);
                var flag = new Rendering.FilledPolygon(new List<float[]>() { new[] { flagx, flagy2 }, new[] { flagx, flagy2 - 10 }, new[] { flagx + 25, flagy2 - 5 } });
                flag.SetColor(.8f, .8f, 0);
                Viewer.AddGeom(flag);
            }

            float pos = State[0];
            CarTrans.SetTranslation((pos - min_position) * scale, Height(pos) * scale);
            CarTrans.SetRotation((float)Math.Cos(3 * pos));

            Viewer.Render();

            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(new[] { Rng.NextFloat(-0.6f, -0.4f), 0.0f}, ObservationSpace.Shape);
            return GetObservation();
        }

        public override bool Step(Tensor action, out Tensor observation, out float reward)
        {
            float position = State[0];
            float velocity = State[1];

            velocity += ((int)action[0] - 1) * 0.001f + (float)Math.Cos(3 * position) * (-0.0025f);
            velocity = Neuro.Tools.Clip(velocity, -max_speed, max_speed);
            position += velocity;
            position = Neuro.Tools.Clip(position, min_position, max_position);
            if (position <= min_position && velocity < 0)
                velocity = 0;

            bool done = position >= goal_position;
            reward = -1.0f;

            State = new Tensor(new[] { position, velocity }, ObservationSpace.Shape);

            observation = GetObservation();
            return done;
        }

        public override void Dispose()
        {
            Viewer.Dispose();
            Viewer = null;
            base.Dispose();
        }

        private float Height(float xs)
        {
            return (float)Math.Sin(3 * xs) * .45f + .55f;
        }

        private Rendering.Viewer Viewer;
        private Rendering.Transform CarTrans;

        private const float min_position = -1.2f;
        private const float max_position = 0.6f;
        private const float max_speed = 0.07f;
        private const float goal_position = 0.5f;
    }
}
