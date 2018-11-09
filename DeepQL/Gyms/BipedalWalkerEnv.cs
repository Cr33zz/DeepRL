using Box2DNet;
using Box2DNet.Dynamics;
using Box2DNet.Common;
using DeepQL.Environments;
using DeepQL.Spaces;
using Neuro.Tensors;
using System;
using System.Linq;
//using System.Numerics;
using System.Collections.Generic;
using DeepQL.Misc;

namespace DeepQL.Gyms
{
    // This is simple 4-joints walker robot environment.
    //
    // There are two versions:
    //
    // - Normal, with slightly uneven terrain.
    //
    // - Hardcore with ladders, stumps, pitfalls.
    //
    // Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
    // it gets -100. Applying motor torque costs a small amount of points, more optimal agent
    // will get better score.
    //
    // Heuristic is provided for testing, it's also useful to get demonstrations to
    // learn from. To run heuristic:
    //
    // python gym/envs/box2d/bipedal_walker.py
    //
    // State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
    // position of joints and joints angular speed, legs contact with ground, and 10 lidar
    // rangefinder measurements to help to deal with the hardcore version. There's no coordinates
    // in the state vector. Lidar is less useful in normal version, but it works.
    //
    // To solve the game you need to get 300 points in 1600 time steps.
    //
    // To solve hardcore version you need 300 points in 2000 time steps.
    //
    // Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
    public class BipedalWalkerEnv : Env
    {
        public BipedalWalkerEnv()
            : base(new Box(new[] { -1.0, -1.0, -1.0, -1.0 }, new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(4)),
                   new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(24)))
        {
            HULL_FD.Vertices = HULL_POLY.Select(v => new Vec2(v[0] / SCALE, v[1] / SCALE)).ToArray();
            HULL_FD.Density = 5.0f;
            HULL_FD.Friction = 0.1f;
            HULL_FD.Filter.CategoryBits = 0x0020;
            HULL_FD.Filter.MaskBits = 0x0001; // collide only with ground
            HULL_FD.Restitution = 0; // 0.99 bouncy

            LEG_FD.SetAsBox(LEG_W / 2, LEG_H / 2);
            LEG_FD.Density = 1.0f;
            LEG_FD.Filter.CategoryBits = 0x0020;
            LEG_FD.Filter.MaskBits = 0x0001;
            LEG_FD.Restitution = 0;

            LOWER_FD.SetAsBox(0.8f * LEG_W / 2, LEG_H / 2);
            LEG_FD.Density = 1.0f;
            LEG_FD.Filter.CategoryBits = 0x0020;
            LEG_FD.Filter.MaskBits = 0x0001;
            LEG_FD.Restitution = 0;

            fd_polygon.Vertices = new Vec2[] { new Vec2(0, 0), new Vec2(1, 0), new Vec2(1, -1), new Vec2(0, -1) };
            fd_polygon.Friction = FRICTION;

            fd_edge.Vertex1 = new Vec2(0, 0);
            fd_edge.Vertex2 = new Vec2(1, 1);
            fd_edge.Friction = FRICTION;
            fd_edge.Filter.CategoryBits = 0x0001;
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            throw new NotImplementedException();
        }

        public override Tensor Reset()
        {
            throw new NotImplementedException();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            throw new NotImplementedException();
        }

        private void GenerateTerrain(bool hardcore)
        {
            const int GRASS = 0, STUMP = 1, STAIRS = 2, PIT = 3, _STATES_ = 4;

            int state = GRASS;
            float velocity = 0.0f;
            float y = TERRAIN_HEIGHT;
            int counter = TERRAIN_STARTPAD;
            bool oneshot = false;
            float original_y = 0;
            terrain.Clear();
            terrain_x.Clear();
            terrain_y.Clear();

            for (int i = 0; i < TERRAIN_LENGTH; ++i)
            {
                float x = i * TERRAIN_STEP;
                terrain_x.Add(x);

                if (state == GRASS && !oneshot)
                {
                    var y_diff = TERRAIN_HEIGHT - y;
                    velocity = 0.8f * velocity + 0.01f * (y_diff > 0 ? 1 : (y_diff == 0 ? 0 : -1));
                    if (i > TERRAIN_STARTPAD)
                        velocity += (float)Rng.NextDouble(-1, 1) / SCALE;   //1
                    y += velocity;
                }
                else if (state == PIT && oneshot)
                {
                    counter = Rng.Next(3, 5);
                    var poly = new Vec2[] {
                        new Vec2(x, y),
                        new Vec2(x + TERRAIN_STEP, y),
                        new Vec2(x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                        new Vec2(x, y - 4 * TERRAIN_STEP)
                    };
                    fd_polygon.Vertices = poly;
                    
                    var t = CreateStaticBody(fd_polygon);
                    //t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    terrain.Add(t);

                    fd_polygon.Vertices = poly.Select(p => new Vec2(p[0] + TERRAIN_STEP * counter, p[1])).ToArray();

                    t = CreateStaticBody(fd_polygon);

                    //t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    terrain.Add(t);
                    counter += 2;
                    original_y = y;
                }
                else if (state == PIT && !oneshot)
                {
                    y = original_y;
                    if (counter > 1)
                        y -= 4 * TERRAIN_STEP;
                }
                /*else if (state== STUMP and oneshot)
                {
                counter = self.np_random.randint(1, 3)
                    poly = [
                        (x, y),
                        (x + counter * TERRAIN_STEP, y),
                        (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                        (x, y + counter * TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                }
                else if (state== STAIRS and oneshot)
                {
                    stair_height = +1 if self.np_random.rand() > 0.5 else -1
                    stair_width = self.np_random.randint(4, 5)
                    stair_steps = self.np_random.randint(3, 5)
                    original_y = y
                    for s in range(stair_steps):
                        poly = [
                            (x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                            (x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                            ]
                        self.fd_polygon.shape.vertices = poly
                        t = self.world.CreateStaticBody(
                            fixtures = self.fd_polygon)
                        t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                        self.terrain.append(t)
                    counter = stair_steps * stair_width
                }
                else if (state == STAIRS and not oneshot)
                {
                    s = stair_steps * stair_width - counter - stair_height
                    n = s / stair_width
                    y = original_y + (n * stair_height) * TERRAIN_STEP
                }

                oneshot = false;
                terrain_y.add(y);
                counter -= 1
                if counter == 0:
                    counter = self.np_random.randint(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                    if state == GRASS and hardcore:
                        state = self.np_random.randint(1, _STATES_)
                        oneshot = True
                    else:
                        state = GRASS
                        oneshot = True*/
            }
            //var terrain_poly = new List<Tuple<>>
            for (int i = 0; i < TERRAIN_LENGTH - 1; ++i)
            {
                /*poly = [
                    (self.terrain_x[i], self.terrain_y[i]),
                    (self.terrain_x[i + 1], self.terrain_y[i + 1])
                    ]
                self.fd_edge.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_edge)
                color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
                t.color1 = color
                t.color2 = color
                self.terrain.append(t)
                color = (0.4, 0.6, 0.3)
                poly += [(poly[1][0], 0), (poly[0][0], 0)]
                self.terrain_poly.append((poly, color))*/
            }

            terrain.Reverse();
        }

        Body CreateStaticBody(FixtureDef fixtureDef)
        {
            var body = World.CreateBody(new BodyDef(1));
            body.CreateFixture(fixtureDef);
            body.SetStatic();
            return body;
        }

        Body CreateDynamicBody(Vec2 position, FixtureDef fixtureDef)
        {
            var body = World.CreateBody(new BodyDef(1));
            body.CreateFixture(fixtureDef);
            body.SetPosition(position);
            return body;
        }

        private const int FPS = 50;
        private const float SCALE = 30.0f;   // affects how fast-paced the game is, forces should be adjusted as well

        private const double MOTORS_TORQUE = 80;
        private const double SPEED_HIP = 4;
        private const double SPEED_KNEE = 6;
        private const double LIDAR_RANGE = 160 / SCALE;

        private const int INITIAL_RANDOM = 5;

        private readonly List<float[]> HULL_POLY = new List<float[]> { new[] { -30.0f, +9 }, new[] { +6.0f, +9 }, new[] { +34.0f, +1 }, new[] { +34.0f, -8 }, new[] { -30.0f, -8 } };

        private const float LEG_DOWN = -8 / SCALE;
        private const float LEG_W = 8 / SCALE;
        private const float LEG_H = 34 / SCALE;

        private const int VIEWPORT_W = 600;
        private const int VIEWPORT_H = 400;

        private const float TERRAIN_STEP = 14 / SCALE;
        private const int TERRAIN_LENGTH = 200;     // in steps
        private const float TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4;
        private const float TERRAIN_GRASS = 10;    // low long are grass spots, in steps
        private const int TERRAIN_STARTPAD = 20;    // in steps
        private const float FRICTION = 2.5f;

        private PolygonDef HULL_FD = new PolygonDef();
        private PolygonDef LEG_FD = new PolygonDef();
        private PolygonDef LOWER_FD = new PolygonDef();

        private Rendering.Viewer Viewer;
        private World World;
        private PolygonDef fd_polygon = new PolygonDef();
        private EdgeDef fd_edge = new EdgeDef();
        private List<Body> terrain;
        private List<float> terrain_x;
        private List<float> terrain_y;
    }
}
