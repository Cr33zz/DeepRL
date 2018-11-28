using Neuro.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using DeepQL.Environments;
using DeepQL.Spaces;
using DeepQL.Misc;

namespace DeepQL.Gyms
{
    /**
    Classic cart-pole system implemented by Rich Sutton et al (implementation based on OpenAI gym https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
    
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    **/
    public class CartPoleEnv : Env
    {
        public CartPoleEnv()
            : base(new Discrete(2), new Box(new float[] { -X_THRESHOLD * 2, float.MinValue, -THETA_THRESHOLD_RADIANS * 2, float.MinValue },
                                            new float[] { X_THRESHOLD * 2, float.MaxValue, THETA_THRESHOLD_RADIANS * 2, float.MaxValue }, new Shape(4)))
        {
            Reset();
        }

        public override bool Step(Tensor action, out Tensor observation, out float reward)
        {
            //Debug.Assert(ActionSpace.Contains(action), "Invalid action");
            float x = State[0];
            float xDot = State[1];
            float theta = State[2];
            float thetaDot = State[3];
            float force = action[0] == 1 ? FORCE_MAG : -FORCE_MAG;
            float cosTheta = (float)Math.Cos(theta);
            float sinTheta = (float)Math.Sin(theta);
            float temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
            float thetaAcc = (GRAVITY * sinTheta - cosTheta * temp) / (LENGTH * (4.0f / 3.0f - MASS_POLE * cosTheta * cosTheta / TOTAL_MASS));
            float xAcc = temp - POLE_MASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

            if (KINEMATICS_INTEGRATOR == EKinematicIntegrator.Euler)
            {
                x = x + TAU * xDot;
                xDot = xDot + TAU * xAcc;
                theta = theta + TAU * thetaDot;
                thetaDot = thetaDot + TAU * thetaAcc;
            }
            //else
            //{
            //    x_dot = x_dot + TAU * xacc;
            //    x = x + TAU * x_dot;
            //    theta_dot = theta_dot + TAU * thetaacc;
            //    theta = theta + TAU * theta_dot;
            //}

            State = new Tensor(new[] { x, xDot, theta, thetaDot }, ObservationSpace.Shape);
            bool done = x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD_RADIANS || theta > THETA_THRESHOLD_RADIANS;

            if (!done)
            {
                reward = 1.0f;
            }
            else if (StepsBeyondDone == -1)
            {
                // Pole just fell!
                StepsBeyondDone = 0;
                reward = 1.0f;
            }
            else
            {
                if (StepsBeyondDone == 0)
                    Console.WriteLine("You are calling 'Step()' even though this environment has already returned done = True. You should always call 'Reset()' once you receive 'done = True' -- any further steps are undefined behavior.");
                ++StepsBeyondDone;
                reward = 0.0f;
            }

            observation = GetObservation();
            return done;
        }

        public override Tensor Reset()
        {
            State = new Tensor(ObservationSpace.Shape);
            State.FillWithRand(-1, -0.05f, 0.05f);
            StepsBeyondDone = -1;
            return GetObservation();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            const int SCREEN_WIDTH = 600;
            const int SCREEN_HEIGHT = 400;

            float worldWidth = X_THRESHOLD * 2;
            float scale = SCREEN_WIDTH / worldWidth;
            float cartY = 100; // TOP OF CART
            float poleWidth = 10.0f;
            float poleLen = scale * (2 * LENGTH);
            float cartWidth = 50.0f;
            float cartHeight = 30.0f;

            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT);
                float l = -cartWidth / 2;
                float r = cartWidth / 2;
                float t = cartHeight / 2;
                float b = -cartHeight / 2;
                float axleOffset = cartHeight / 4.0f;
                var cart = new Rendering.FilledPolygon(new List<float[]> { new[] { l, b }, new[] { l, t }, new[] { r, t }, new[] { r, b } });
                CartTrans = new Rendering.Transform();
                cart.AddAttr(CartTrans);
                Viewer.AddGeom(cart);
                l = -poleWidth / 2;
                r = poleWidth / 2;
                t = poleLen - poleWidth / 2;
                b = -poleWidth / 2;
                Pole = new Rendering.FilledPolygon(new List<float[]> { new[] { l, b }, new[] { l, t }, new[] { r, t }, new[] { r, b } });
                Pole.SetColor(.8f, .6f, .4f);
                PoleTrans = new Rendering.Transform(new[] {0, axleOffset});
                Pole.AddAttr(PoleTrans);
                Pole.AddAttr(CartTrans);
                Viewer.AddGeom(Pole);
                Axle = Rendering.MakeCircle(poleWidth / 2);
                Axle.AddAttr(PoleTrans);
                Axle.AddAttr(CartTrans);
                Axle.SetColor(.5f, .5f, .8f);
                Viewer.AddGeom(Axle);
                Track = new Rendering.Line(new [] { 0, cartY }, new [] { SCREEN_WIDTH, cartY });
                Track.SetColor(0, 0, 0);
                Viewer.AddGeom(Track);
            }

            if (State == null)
                return null;

            {
                // Edit the pole polygon vertex
                float l = -poleWidth / 2, r = poleWidth / 2, t = poleLen - poleWidth / 2, b = poleWidth / 2;
                Pole.Vertices = new List<float[]> {new[] {l, b}, new[] {l, t}, new[] {r, t}, new[] {r, b}};
            }

            var cartX = State[0] * scale + SCREEN_WIDTH / 2.0f; // MIDDLE OF CART
            CartTrans.SetTranslation(cartX, cartY);
            PoleTrans.SetRotation(-State[2]);

            byte[] rgbArray = toRgbArray ? new byte[SCREEN_WIDTH * SCREEN_HEIGHT * 3] : null;
            Viewer.Render(rgbArray);
            return rgbArray;
        }

        public override void Dispose()
        {
            Viewer.Dispose();
            Viewer = null;
            base.Dispose();
        }

        private enum EKinematicIntegrator
        {
            Euler,
            SemiImplicitEuler,
        }

        private const float GRAVITY = 9.8f;
        private const float MASS_CART = 1.0f;
        private const float MASS_POLE = 0.1f;
        private const float TOTAL_MASS = (MASS_POLE + MASS_CART);
        private const float LENGTH = 0.5f; // actually half the pole's length
        private const float POLE_MASS_LENGTH = (MASS_POLE * LENGTH);
        private const float FORCE_MAG = 10.0f;
        private const float TAU = 0.02f;  // seconds between state updates
        private const EKinematicIntegrator KINEMATICS_INTEGRATOR = EKinematicIntegrator.Euler;

        // Angle at which to fail the episode
        private const float THETA_THRESHOLD_RADIANS = 12 * 2 * (float)Math.PI / 360;
        private const float X_THRESHOLD = 2.4f;

        private Rendering.Viewer Viewer;
        private Rendering.FilledPolygon Pole;
        private Rendering.Geom Axle;
        private Rendering.Geom Track;
        private Rendering.Transform CartTrans;
        private Rendering.Transform PoleTrans;

        private int StepsBeyondDone = -1;
    }
}
