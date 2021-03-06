﻿using DeepQL.Environments;
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
                   new Box(new [] { -1, -1, -max_speed },
                           new [] { 1, 1, max_speed }, new Shape(3)))
        {
            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(500, 500);
                Viewer.SetBounds(-2.2f, 2.2f, -2.2f, 2.2f);
                var rod = Rendering.MakeCapsule(1, .2f);
                rod.SetColor(.8f, .3f, .3f);
                pole_transform = new Rendering.Transform();
                rod.AddAttr(pole_transform);
                Viewer.AddGeom(rod);
                var axle = Rendering.MakeCircle(.05f);
                axle.SetColor(0, 0, 0);
                Viewer.AddGeom(axle);
                //img = new Rendering.Image("assets/clockwise.png", 1.0, 1.0);
                //imgtrans = new Rendering.Transform();
                //img.AddAttr(imgtrans);
            }

            pole_transform.SetRotation(State[0] + (float)Math.PI / 2);

            //Viewer.AddOneTime(img);
            //if (!float.IsNaN(last_u))
            //    imgtrans.SetScale(-last_u / 2, Math.Abs(last_u) / 2 );

            Viewer.Render();

            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(new Shape(2));
            State[0] = Rng.NextFloat(-(float)Math.PI, (float)Math.PI);
            State[1] = Rng.NextFloat(-1, 1);
            last_u = float.NaN;
            return GetObservation();
        }

        public override bool Step(Tensor action, out Tensor observation, out float reward)
        {
            float th = State[0], thdot = State[1]; // th := theta

            float g = 10.0f;
            float m = 1.0f;
            float l = 1.0f;

            float u = action[0];
            u = Neuro.Tools.Clip(u, -max_torque, max_torque);
            last_u = u; // for rendering
            float costs = (float)Math.Pow(AngleNormalize(th), 2) + 0.1f * (float)Math.Pow(thdot, 2) + .001f * (float)Math.Pow(u, 2);

            float newthdot = thdot + (-3 * g / (2 * l) * (float)Math.Sin(th + Math.PI) + 3.0f / (m * (float)Math.Pow(l, 2)) * u) * dt;
            float newth = th + newthdot * dt;
            newthdot = Neuro.Tools.Clip(newthdot, -max_speed, max_speed); //pylint: disable=E1111

            State = new Tensor(new[] { newth, newthdot }, State.Shape);
            observation = GetObservation();
            reward = -costs;

            return false;
        }

        public override void Dispose()
        {
            Viewer.Dispose();
            Viewer = null;
            base.Dispose();
        }

        protected override Tensor GetObservation()
        {
            float theta = State[0], thetadot = State[1];
            return new Tensor(new[] { (float)Math.Cos(theta), (float)Math.Sin(theta), thetadot }, ObservationSpace.Shape);
        }

        private float AngleNormalize(float x)
        {
            return (float)(((x + Math.PI) % (2 * Math.PI)) - Math.PI);
        }

        private Rendering.Viewer Viewer;
        //private Rendering.Image img;
        //private Rendering.Transform imgtrans;
        private Rendering.Transform pole_transform;
        private float last_u = float.NaN;

        private const float max_speed = 8;
        private const float max_torque = 2.0f;
        private const float dt = .05f;
    }
}
