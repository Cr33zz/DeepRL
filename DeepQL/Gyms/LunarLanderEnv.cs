using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DeepQL.Environments;
using DeepQL.Misc;
using DeepQL.Spaces;
using Neuro.Tensors;
using Enum = Mono.CSharp.Enum;

namespace DeepQL.Gyms
{
    // Rocket trajectory optimization is a classic topic in Optimal Control.
    //
    // According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
    // turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
    //
    // Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
    // Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
    // If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
    // comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
    // engine is -0.3 points each frame. Solved is 200 points.
    //
    // Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
    // on its first attempt. Please see source code for details.
    //
    // Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
    public class LunarLanderEnv : Env
    {
        public LunarLanderEnv()
            : this(false)
        {
        }

        protected LunarLanderEnv(bool continuous = false)
            : base(null, null)
        {
            this.continuous = continuous;

            world = new b2World(new b2Vec2(0, -10));

            // useful range is -1 .. +1, but spikes can be higher
            ObservationSpace = new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(8));

            if (continuous)
            {
                // Action is two floats [main engine, left-right engines].
                // Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
                // Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
                ActionSpace = new Box(-1, 1, new Shape(2));
            }
            else
            {
                // Nop, fire left engine, main engine, right engine
                ActionSpace = new Discrete(4);
            }

            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(VIEWPORT_W, VIEWPORT_H);
                Viewer.SetBounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE);
            }

            foreach (var obj in particles)
            {
                var data = obj.GetUserData() as CustomBodyData;
                data.TimeToLive -= 0.15f;
                data.Color1 = new b2Vec3((float)Math.Max(0.2, 0.2 + data.TimeToLive), (float)Math.Max(0.2, 0.5 * data.TimeToLive), (float)Math.Max(0.2, 0.5 * data.TimeToLive));
                data.Color2 = new b2Vec3((float)Math.Max(0.2, 0.2 + data.TimeToLive), (float)Math.Max(0.2, 0.5 * data.TimeToLive), (float)Math.Max(0.2, 0.5 * data.TimeToLive));
            }

            CleanParticles(false);

            foreach (var p in sky_polys)
                Viewer.DrawPolygon(p).SetColor(0, 0, 0);

            foreach (var obj in particles.Concat(drawlist))
            {
                for (b2Fixture f = obj.GetFixtureList(); f != null; f = f.GetNext())
                {
                    var xform = f.GetBody().GetTransform();
                    if (f.GetShape() is b2CircleShape)
                    {
                        var shape = (f.GetShape() as b2CircleShape);
                        var customData = (obj.GetUserData() as CustomBodyData);
                        var trans = Utils.b2Mul(xform, shape.m_p);

                        var t = new Rendering.Transform(new double[] { trans[0], trans[1] });
                        Viewer.DrawCircle(shape.m_radius, 20).SetColor(customData.Color1.x, customData.Color1.y, customData.Color1.z).AddAttr(t);
                        Viewer.DrawCircle(shape.m_radius, 20, false).SetColor(customData.Color2.x, customData.Color2.y, customData.Color2.z).SetLineWidth(2).AddAttr(t);
                    }
                    else
                    {
                        var shape = f.GetShape();
                        var customData = (obj.GetUserData() as CustomBodyData);

                        var path = shape.GetVertices().Select(v => { var trans = Utils.b2Mul(xform, v); return new double[] { trans[0], trans[1] }; }).ToList();
                        Viewer.DrawPolygon(path).SetColor(customData.Color1.x, customData.Color1.y, customData.Color1.z);
                        path.Add(path[0]);
                        Viewer.DrawPolyline(path).SetColor(customData.Color2.x, customData.Color2.y, customData.Color2.z).SetLineWidth(2);
                    }
                }
            }


            foreach (var x in new[] {helipad_x1, helipad_x2})
            {
                var flagy1 = helipad_y;
                var flagy2 = flagy1 + 50 / SCALE;
                Viewer.DrawPolyline(new List<double[]> { new double[] {x, flagy1}, new double[] {x, flagy2}}).SetColor(1, 1, 1);
                Viewer.DrawPolygon(new List<double[]> { new double[] {x, flagy2}, new double[] {x, flagy2-10/SCALE}, new double[] {x+25/SCALE, flagy2-5/SCALE}}).SetColor(0.8, 0.8, 0);
            }

            Viewer.Render();
            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(ObservationSpace.Shape);

            Destroy();
            contact_detector = new ContactDetector(this);
            world.SetContactListener(contact_detector);
            game_over = false;
            prev_shaping = float.NaN;

            var W = VIEWPORT_W / SCALE;
            var H = VIEWPORT_H / SCALE;

            // terrain
            var CHUNKS = 11;
            var height = Rng.NextFloat(0.0f, H / 2.0f, CHUNKS + 1);
            var chunk_x = Enumerable.Range(0, CHUNKS).Select(i => W / (CHUNKS - 1) * i).ToArray();

            helipad_x1 = chunk_x[(int)Math.Floor(CHUNKS / 2.0) - 1];
            helipad_x2 = chunk_x[(int)Math.Floor(CHUNKS / 2.0) + 1];
            helipad_y = H / 4;
            height[(int)Math.Floor(CHUNKS / 2.0) - 2] = helipad_y;
            height[(int)Math.Floor(CHUNKS / 2.0) - 1] = helipad_y;
            height[(int)Math.Floor(CHUNKS / 2.0) + 0] = helipad_y;
            height[(int)Math.Floor(CHUNKS / 2.0) + 1] = helipad_y;
            height[(int)Math.Floor(CHUNKS / 2.0) + 2] = helipad_y;
            var smooth_y = Enumerable.Range(0, CHUNKS).Select(i => 0.33f * (height[(CHUNKS + i - 1)%CHUNKS] + height[i + 0] + height[i + 1])).ToArray();

            moon = world.CreateStaticBody(new b2FixtureDef(){ shape = new b2EdgeShape().Set(new b2Vec2(0, 0), new b2Vec2(W, 0)) });
            sky_polys.Clear();
            for (int i = 0; i < CHUNKS - 1; ++i)
            {
                var p1 = new double[] {chunk_x[i], smooth_y[i]};
                var p2 = new double[] {chunk_x[i + 1], smooth_y[i + 1]};
                moon.CreateFixture(new b2FixtureDef(){shape = new b2EdgeShape().Set(new b2Vec2((float)p1[0], (float)p1[1]), new b2Vec2((float)p2[0], (float)p2[1])), density = 0, friction = 0.1f});
                sky_polys.Add(new List<double[]>{ p1, p2, new[] {p2[0], H}, new[] {p1[0], H}} );
            }

            moon.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0,0,0), Color2 = new b2Vec3(0, 0, 0) });

            var initial_y = VIEWPORT_H / SCALE;
            lander = world.CreateDynamicBody(new b2Vec2(VIEWPORT_W / SCALE / 2, initial_y), 
                                             0.0f,
                                             new b2FixtureDef() {
                                                 shape = new b2PolygonShape().Set(LANDER_POLY.Select(p => new b2Vec2(p[0] / SCALE, p[1] / SCALE)).ToArray()),
                                                 density = 5.0f,
                                                 friction = 0.1f,
                                                 filter = new b2Filter() {categoryBits = 0x0010, maskBits = 0x001 }, // collide only with ground
                                                 restitution = 0.0f} // 0.99 bouncy
                                             );
            lander.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0.5f, 0.4f, 0.9f), Color2 = new b2Vec3(0.3f, 0.3f, 0.5f) });
            lander.ApplyForceToCenter(new b2Vec2((float)Rng.NextDouble(-INITIAL_RANDOM, INITIAL_RANDOM), (float)Rng.NextDouble(-INITIAL_RANDOM, INITIAL_RANDOM)), true);

            legs.Clear();
            foreach (int i in new[] {-1, +1})
            {
                var leg = world.CreateDynamicBody(
                    new b2Vec2(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                    i * 0.05f,
                    new b2FixtureDef()
                    {
                        shape = new b2PolygonShape().SetAsBox(LEG_W / SCALE, LEG_H / SCALE),
                        density = 1.0f,
                        restitution = 0.0f,
                        filter = new b2Filter() {categoryBits = 0x0020, maskBits = 0x001}
                    }
                );
                leg.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0.5f, 0.4f, 0.9f), Color2 = new b2Vec3(0.3f, 0.3f, 0.5f) });

                var rjd = new b2RevoluteJointDef()
                {
                    bodyA = lander,
                    bodyB = leg,
                    localAnchorA = new b2Vec2(0, 0),
                    localAnchorB = new b2Vec2(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                    enableMotor = true,
                    enableLimit = true,
                    maxMotorTorque = LEG_SPRING_TORQUE,
                    motorSpeed = +0.3f * i
                    // low enough not to jump back into the sky
                };
                if (i == -1)
                {
                    rjd.lowerAngle = +0.9f - 0.5f;
                    // Yes, the most esoteric numbers here, angles legs have freedom to travel within
                    rjd.upperAngle = +0.9f;
                }
                else
                {
                    rjd.lowerAngle = -0.9f;
                    rjd.upperAngle = -0.9f + 0.5f;
                }

                world.CreateJoint(rjd);
                legs.Add(leg);
            }

            drawlist = new List<b2Body>{lander};
            drawlist = drawlist.Concat(legs).ToList();

            Step(continuous ? new Tensor(new[] { 0.0, 0.0}, new Shape(2)) : new Tensor(new[] { 0.0 }, new Shape(1)), out var observation, out var reward);

            return observation;
        }

        private b2Body CreateParticle(float mass, float x, float y, float ttl)
        {
            var p = world.CreateDynamicBody(
                new b2Vec2(x, y),
                0.0f,
                new b2FixtureDef()
                {
                    shape = new b2CircleShape() {m_radius = 2 / SCALE, m_p = new b2Vec2(0, 0)},
                    density = mass,
                    friction = 0.1f,
                    filter = new b2Filter() {categoryBits = 0x0100, maskBits = 0x001}, // collide only with ground
                    restitution = 0.3f
                }
            );
            p.SetUserData(new CustomBodyData() {TimeToLive = ttl});
            particles.Add(p);
            CleanParticles(false);
            return p;
        }

        private void CleanParticles(bool all)
        {
            while (particles.Count > 0 && (all || (particles[0].GetUserData() as CustomBodyData).TimeToLive < 0))
            {
                world.DestroyBody(particles.First());
                particles.RemoveAt(0);
            }
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            if (continuous)
                action = action.Clipped(-1, +1);
            //else
            //    Assert(ActionSpace.Contains(action));

            // Engines
            var tip = new[] {(float)Math.Sin(lander.GetAngle()), (float)Math.Cos(lander.GetAngle())};
            var side = new[] {-tip[1], tip[0]};
            var dispersion = Enumerable.Range(0, 2).Select(i => (float)Rng.NextDouble(-1.0, +1.0) / SCALE).ToArray();

            var m_power = 0.0f;
            if ((continuous && action[0] > 0.0) || (!continuous && action[0] == 2))
            {
                // Main engine
                if (continuous)
                {
                    m_power = (float) (Neuro.Tools.Clip(action[0], 0.0, 1.0) + 1.0) * 0.5f; // 0.5..1.0;
                    Debug.Assert(m_power >= 0.5f && m_power <= 1.0f);
                }
                else
                    m_power = 1.0f;

                var ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]; // 4 is move a bit downwards, +-2 for randomness
                var oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1];
                var impulse_pos = new b2Vec2(lander.GetPosition()[0] + ox, lander.GetPosition()[1] + oy);
                var p = CreateParticle(3.5f, impulse_pos[0], impulse_pos[1], m_power); // particles are just a decoration, 3.5 is here to make particle speed adequate
                p.ApplyLinearImpulse(new b2Vec2(ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, true);
                lander.ApplyLinearImpulse(new b2Vec2(-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, true);
            }

            var s_power = 0.0f;
            var direction = 0;
            if ((continuous && Math.Abs(action[1]) > 0.5) || (!continuous && (action[0] == 1 || action[0] == 3)))
            {
            // Orientation engines
                if (continuous)
                {
                    direction = Neuro.Tools.Sign(action[1]);
                    s_power = (float)Neuro.Tools.Clip(Math.Abs(action[1]), 0.5, 1.0);
                    Debug.Assert(s_power >= 0.5 && s_power <= 1.0);
                }
                else
                {
                    direction = (int)action[0] - 2;
                    s_power = 1.0f;
                }

                var ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
                var oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
                var impulse_pos = new b2Vec2(lander.GetPosition()[0] + ox - tip[0] * 17 / SCALE,
                                             lander.GetPosition()[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE);
                var p = CreateParticle(0.7f, impulse_pos[0], impulse_pos[1], s_power);
                p.ApplyLinearImpulse(new b2Vec2(ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, true);
                lander.ApplyLinearImpulse(new b2Vec2(-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, true);
            }

            world.Step(1.0f / FPS, 6 * 30, 2 * 30);

            var pos = lander.GetPosition();
            var vel = lander.GetLinearVelocity();

            var state = new double[]
            {
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                lander.GetAngle(),
                20.0 * lander.GetAngularVelocity() / FPS,
                (legs[0].GetUserData() as CustomBodyData).GroundContact ? 1.0 : 0.0,
                (legs[1].GetUserData() as CustomBodyData).GroundContact ? 1.0 : 0.0
            };
            Debug.Assert(state.Length == 8);

            reward = 0;
            var shaping = -100 * Math.Sqrt(state[0] * state[0] + state[1] * state[1])
                          - 100 * Math.Sqrt(state[2] * state[2] + state[3] * state[3])
                          - 100 * Math.Abs(state[4]) + 10 * state[6] + 10 * state[7];   // And ten points for legs contact, the idea is if you
                                                                                        // lose contact again after landing, you get negative reward
            if (!float.IsNaN(prev_shaping))
                reward = shaping - prev_shaping;
            prev_shaping = (float)shaping;

            reward -= m_power * 0.30f;  // less fuel spent is better, about -30 for heurisic landing
            reward -= s_power * 0.03f;

            var done = false;
            if (game_over || Math.Abs(state[0]) >= 1.0)
            {
                done = true;
                reward = -100;
            }
            if (!lander.IsAwake())
            {
                done = true;
                reward = +100;
            }

            State = new Tensor(state, State.Shape);
            observation = GetObservation();
            return done;
        }

        private void Destroy()
        {
            if (moon == null) return;
            world.SetContactListener(null);
            CleanParticles(true);
            world.DestroyBody(moon);
            moon = null;
            world.DestroyBody(lander);
            lander = null;
            world.DestroyBody(legs[0]);
            world.DestroyBody(legs[1]);
        }

    private class ContactDetector : b2ContactListener
        {
            public ContactDetector(LunarLanderEnv env)
            {
                Env = env;
            }

            public override void BeginContact(b2Contact contact)
            {
                if (Env.lander == contact.GetFixtureA().GetBody() || Env.lander == contact.GetFixtureB().GetBody())
                    Env.game_over = true;
                foreach (var leg in Env.legs)
                    if (leg == contact.GetFixtureA().GetBody() || leg == contact.GetFixtureB().GetBody())
                        (leg.GetUserData() as CustomBodyData).GroundContact = true;
            }

            public override void EndContact(b2Contact contact)
            {
                foreach (var leg in Env.legs)
                    if (leg == contact.GetFixtureA().GetBody() || leg == contact.GetFixtureB().GetBody())
                        (leg.GetUserData() as CustomBodyData).GroundContact = false;
            }

            private LunarLanderEnv Env;
        }

        private class CustomBodyData
        {
            public b2Vec3 Color1;
            public b2Vec3 Color2;
            public bool GroundContact;
            public float TimeToLive;
        }

        private Rendering.Viewer Viewer;
        private b2World world;
        private b2Body moon;
        private List<b2Body> legs = new List<b2Body>();
        private List<b2Body> particles = new List<b2Body>();
        private b2Body lander;
        private bool game_over;
        //private double prev_reward;
        private float prev_shaping;
        private float helipad_x1;
        private float helipad_x2;
        private float helipad_y;
        private readonly List<List<double[]>> sky_polys = new List<List<double[]>>();
        private List<b2Body> drawlist = new List<b2Body>();
        private ContactDetector contact_detector;

        private readonly bool continuous;

        private const int FPS = 50;
        private const float SCALE = 30.0f;   // affects how fast-paced the game is, forces should be adjusted as well

        private const float MAIN_ENGINE_POWER = 13.0f;
        private const float SIDE_ENGINE_POWER = 0.6f;

        private const float INITIAL_RANDOM = 1000.0f;   // Set 1500 to make game harder

        private readonly List<float[]> LANDER_POLY = new List<float[]> { new[] {-14.0f, +17}, new[] {-17.0f, 0}, new[] {-17.0f, -10}, new[] {+17.0f, -10}, new[] {+17.0f, 0}, new[] {+14.0f, +17} };
        private const float LEG_AWAY = 20;
        private const float LEG_DOWN = 18;
        private const float LEG_W = 2;
        private const float LEG_H = 8;
        private const float LEG_SPRING_TORQUE = 40;

        private const float SIDE_ENGINE_HEIGHT = 14.0f;
        private const float SIDE_ENGINE_AWAY = 12.0f;

        private const int VIEWPORT_W = 600;
        private const int VIEWPORT_H = 400;
    }

    public class LunarLanderContinuousEnv : LunarLanderEnv
    {
        public LunarLanderContinuousEnv() : base(true) { }
    }
}
