using DeepQL.Environments;
using Neuro.Tensors;
using System;
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
            : base(new Discrete(2), new Box(new double[] { -x_threshold * 2, double.MinValue, -theta_threshold_radians * 2, double.MinValue },
                                            new double[] { x_threshold * 2, double.MaxValue, theta_threshold_radians * 2, double.MaxValue }, new Shape(4)))
        {
            Seed();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            //Debug.Assert(ActionSpace.Contains(action), "Invalid action");
            double x = State[0];
            double x_dot = State[1];
            double theta = State[2];
            double theta_dot = State[3];
            double force = action[0] == 1 ? force_mag : -force_mag;
            double costheta = Math.Cos(theta);
            double sintheta = Math.Sin(theta);
            double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
            double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
            double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

            if (kinematics_integrator == EKinematicIntegrator.Euler)
            {
                x = x + tau * x_dot;
                x_dot = x_dot + tau * xacc;
                theta = theta + tau * theta_dot;
                theta_dot = theta_dot + tau * thetaacc;
            }
            else
            {
                x_dot = x_dot + tau * xacc;
                x = x + tau * x_dot;
                theta_dot = theta_dot + tau * thetaacc;
                theta = theta + tau * theta_dot;
            }

            State = new Tensor(new[] { x, x_dot, theta, theta_dot }, ObservationSpace.Shape);
            bool done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;

            if (!done)
            {
                reward = 1.0;
            }
            else if (steps_beyond_done == -1)
            {
                // Pole just fell!
                steps_beyond_done = 0;
                reward = 1.0;
            }
            else
            {
                if (steps_beyond_done == 0)
                    Console.WriteLine("You are calling 'Step()' even though this environment has already returned done = True. You should always call 'Reset()' once you receive 'done = True' -- any further steps are undefined behavior.");
                steps_beyond_done += 1;
                reward = 0.0;
            }

            observation = State.Clone();
            return done;
        }

        public override Tensor Reset()
        {
            throw new NotImplementedException();
        }

        public override void Render()
        {
            throw new NotImplementedException();
        }

        public override void Dispose()
        {
            throw new NotImplementedException();
        }

        private enum EKinematicIntegrator
        {
            Euler,
            SemiImplicitEuler,
        }

        private const double gravity = 9.8;
        private const double masscart = 1.0;
        private const double masspole = 0.1;
        private const double total_mass = (masspole + masscart);
        private const double length = 0.5; // actually half the pole's length
        private const double polemass_length = (masspole * length);
        private const double force_mag = 10.0;
        private const double tau = 0.02;  // seconds between state updates
        private const EKinematicIntegrator kinematics_integrator = EKinematicIntegrator.Euler;

        // Angle at which to fail the episode
        private const double theta_threshold_radians = 12 * 2 * Math.PI / 360;
        private const double x_threshold = 2.4;
        
        Viewer viewer;

        int steps_beyond_done = -1;
    }
}
