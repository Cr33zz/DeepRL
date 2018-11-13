using DeepQL.Environments;
using DeepQL.Spaces;
using Neuro.Tensors;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using DeepQL.Misc;
using Shape = Neuro.Tensors.Shape;

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
            contactDetector = new ContactDetector(this);
            var worldAabb = new b2AABB() { lowerBound = new b2Vec2(-100, -100), upperBound = new b2Vec2(1000, 1000) };
            World = new b2World(new b2Vec2(0, 9.807f));

            HULL_FD.shape = new b2PolygonShape();
            (HULL_FD.shape as b2PolygonShape).Set(HULL_POLY.Select(v => new b2Vec2(v[0] / SCALE, v[1] / SCALE)).ToArray());
            HULL_FD.density = 5.0f;
            HULL_FD.friction = 0.1f;
            HULL_FD.filter.categoryBits = 0x0020;
            HULL_FD.filter.maskBits = 0x0001; // collide only with ground
            HULL_FD.restitution = 0; // 0.99 bouncy

            LEG_FD.shape = new b2PolygonShape();
            (LEG_FD.shape as b2PolygonShape).SetAsBox(LEG_W / 2, LEG_H / 2);
            LEG_FD.density = 1.0f;
            LEG_FD.filter.categoryBits = 0x0020;
            LEG_FD.filter.maskBits = 0x0001;
            LEG_FD.restitution = 0;

            LOWER_FD.shape = new b2PolygonShape();
            (LOWER_FD.shape as b2PolygonShape).SetAsBox(0.8f * LEG_W / 2, LEG_H / 2);
            LEG_FD.density = 1.0f;
            LEG_FD.filter.categoryBits = 0x0020;
            LEG_FD.filter.maskBits = 0x0001;
            LEG_FD.restitution = 0;

            fd_polygon.shape = new b2PolygonShape();
            (fd_polygon.shape as b2PolygonShape).Set(new[] { new b2Vec2(0, 0), new b2Vec2(1, 0), new b2Vec2(1, -1), new b2Vec2(0, -1) });
            fd_polygon.friction = FRICTION;

            fd_edge.shape = new b2EdgeShape();
            (fd_edge.shape as b2EdgeShape).Set(new b2Vec2(0, 0), new b2Vec2(1, 1));
            fd_edge.friction = FRICTION;
            fd_edge.filter.categoryBits = 0x0001;

            Reset();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
                Viewer = new Rendering.Viewer(VIEWPORT_W, VIEWPORT_H);
            Viewer.SetBounds(scroll, VIEWPORT_W / SCALE + scroll, 0, VIEWPORT_H / SCALE);

            Viewer.DrawPolygon(new List<double[]>{ new double[] {scroll, 0},
                                                   new double[] {scroll + VIEWPORT_W / SCALE, 0},
                                                   new double[] {scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE},
                                                   new double[] {scroll, VIEWPORT_H / SCALE} }).SetColor(0.9, 0.9, 1.0);

            //for poly, x1, x2 in self.cloud_poly:
            //    if x2 < self.scroll / 2: continue
            //    if x1 > self.scroll / 2 + VIEWPORT_W / SCALE: continue
            //    self.viewer.draw_polygon( [(p[0] + self.scroll / 2, p[1]) for p in poly], color = (1, 1, 1))
            //        for poly, color in self.terrain_poly:
            //    if poly[1][0] < self.scroll: continue
            //    if poly[0][0] > self.scroll + VIEWPORT_W / SCALE: continue
            //    self.viewer.draw_polygon(poly, color = color)

            foreach (var l in lidar)
            {
                Viewer.DrawPolyline(new List<double[]>{ new double[]{l.P1.x, l.P1.y}, new double[] { l.P2.x, l.P2.y} }).SetColor(1, 0, 0).SetLineWidth(1);
            }

            foreach (var obj in drawlist)
            {
                for (b2Fixture f = obj.GetFixtureList(); f != null; f = f.GetNext())
                {
                    var xform = f.GetBody().GetTransform();

                    if (f.GetShape() is b2CircleShape)
                    {
                        var shape = (f.GetShape() as b2CircleShape);
                        var customData = (obj.GetUserData() as CustomBodyData);
                        var trans = obj.GetPosition() + GlobalMembers.b2Mul(xform, shape.m_p);

                        var t = new Rendering.Transform(new double[] { trans[0], trans[1] });
                        Viewer.DrawCircle(shape.m_radius, 30).SetColor(customData.Color1.x, customData.Color1.y, customData.Color1.z).AddAttr(t);
                        Viewer.DrawCircle(shape.m_radius, 30, false).SetColor(customData.Color2.x, customData.Color2.y, customData.Color2.z).SetLineWidth(2).AddAttr(t);
                    }
                    else if (f.GetShape() is b2PolygonShape)
                    {
                        var shape = (f.GetShape() as b2PolygonShape);
                        var customData = (obj.GetUserData() as CustomBodyData);

                        var path = shape.m_vertices.Select(v => { var trans = obj.GetPosition() + GlobalMembers.b2Mul(xform, v); return new double[] { trans[0], trans[1] };}).ToList();
                        Viewer.DrawPolygon(path).SetColor(customData.Color1.x, customData.Color1.y, customData.Color1.z);
                        path.Add(path[0]);
                        Viewer.DrawPolyline(path).SetColor(customData.Color2.x, customData.Color2.y, customData.Color2.z).SetLineWidth(2);
                    }
                }
            }

            var flagy1 = TERRAIN_HEIGHT;
            var flagy2 = flagy1 + 50 / SCALE;
            var x = TERRAIN_STEP * 3;

            Viewer.DrawPolyline(new List<double[]>{new double[] {x, flagy1}, new double[] { x, flagy2 }}).SetColor(0,0,0).SetLineWidth(2);
            var flagVert = new List<double[]>{ new double[] { x, flagy2 }, new double[] {x, flagy2 - 10 / SCALE}, new double[] { x + 25 / SCALE, flagy2 - 5 / SCALE }};
            Viewer.DrawPolygon(flagVert).SetColor(0.9, 0.2, 0);
            flagVert.Add(flagVert[0]);
            Viewer.DrawPolyline(flagVert).SetColor(0, 0, 0).SetLineWidth(2);

            Viewer.Render();
            return null;
        }

        public override Tensor Reset()
        {
            State = new Tensor(ObservationSpace.Shape);

            Destroy();
            World.SetContactListener(contactDetector);

            game_over = false;
            prev_shaping = float.NaN;
            scroll = 0.0f;
            lidar_render = 0;

            var W = VIEWPORT_W / SCALE;
            var H = VIEWPORT_H / SCALE;

            GenerateTerrain(hardcore);
            //GenerateClouds();

            var init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2;
            var init_y = TERRAIN_HEIGHT + 2 * LEG_H;
            hull = CreateDynamicBody(new b2Vec2(init_x, init_y), 0, HULL_FD);
            hull.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0.5f, 0.4f, 0.9f), Color2 = new b2Vec3(0.3f, 0.3f, 0.5f) });
            hull.ApplyForce(new b2Vec2((float)Rng.NextDouble(-INITIAL_RANDOM, INITIAL_RANDOM), 0), hull.GetWorldCenter(), true);

            foreach (var i in new[] {-1, +1})
            {
                var leg = CreateDynamicBody(new b2Vec2(init_x, init_y - LEG_H / 2 - LEG_DOWN), (i * 0.05f), LEG_FD);
                leg.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0.6f - i / 10.0f, 0.3f - i / 10.0f, 0.5f - i / 10.0f),
                                                       Color2 = new b2Vec3(0.4f - i / 10.0f, 0.2f - i / 10.0f, 0.3f - i / 10.0f) });
                var rjd = new b2RevoluteJointDef();
                rjd.bodyA = hull;
                rjd.bodyB = leg;
                rjd.localAnchorA = new b2Vec2(0, LEG_DOWN);
                rjd.localAnchorB = new b2Vec2(0, LEG_H / 2);
                rjd.enableMotor = true;
                rjd.enableLimit = true;
                rjd.maxMotorTorque = MOTORS_TORQUE;
                rjd.motorSpeed = i;
                rjd.lowerAngle = -0.8f;
                rjd.upperAngle = 1.1f;

                legs.Add(leg);
                joints.Add((b2RevoluteJoint)World.CreateJoint(rjd));

                var lower = CreateDynamicBody(new b2Vec2(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN), (i * 0.05f), LOWER_FD);
                lower.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(0.6f - i / 10.0f, 0.3f - i / 10.0f, 0.5f - i / 10.0f),
                                                        Color2 = new b2Vec3(0.4f - i / 10.0f, 0.2f - i / 10.0f, 0.3f - i / 10.0f) });
                rjd = new b2RevoluteJointDef();
                rjd.bodyA = leg;
                rjd.bodyB = lower;
                rjd.localAnchorA = new b2Vec2(0, -LEG_H / 2);
                rjd.localAnchorB = new b2Vec2(0, LEG_H / 2);
                rjd.enableMotor = true;
                rjd.enableLimit = true;
                rjd.maxMotorTorque = MOTORS_TORQUE;
                rjd.motorSpeed = 1;
                rjd.lowerAngle = -1.6f;
                rjd.upperAngle = -0.1f;
                
                legs.Add(lower);
                joints.Add((b2RevoluteJoint)World.CreateJoint(rjd));
            }

            drawlist = terrain.Concat(legs).ToList();
            drawlist.Add(hull);

            for (int i = 0; i < lidar.Length; ++i)
                lidar[i] = new LidarCallback();

            Step(new Tensor(new[] {0.0, 0.0, 0.0, 0.0}, ActionSpace.Shape), out var observation, out var reward);

            return observation;

        }

        private class LidarCallback : b2RayCastCallback
        {
            public b2Vec2 P1, P2;
            public double Fraction;

            public override float ReportFixture(b2Fixture fixture, b2Vec2 point, b2Vec2 normal, float fraction)
            {
                if ((fixture.GetFilterData().categoryBits & 1) == 0)
                    return 1;

                P2 = point;
                Fraction = fraction;
                return 0;
            }
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            // hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
            var control_speed = false;  // Should be easier as well
            if (control_speed)
            {
                joints[0].SetMotorSpeed((float)(SPEED_HIP * Neuro.Tools.Clip(action[0], -1, 1)));
                joints[1].SetMotorSpeed((float)(SPEED_KNEE * Neuro.Tools.Clip(action[1], -1, 1)));
                joints[2].SetMotorSpeed((float)(SPEED_HIP * Neuro.Tools.Clip(action[2], -1, 1)));
                joints[3].SetMotorSpeed((float)(SPEED_KNEE * Neuro.Tools.Clip(action[3], -1, 1)));
            }
            else
            {
                joints[0].SetMotorSpeed((SPEED_HIP * Neuro.Tools.Sign(action[0])));
                joints[0].SetMaxMotorTorque((float)(MOTORS_TORQUE * Neuro.Tools.Clip(System.Math.Abs(action[0]), 0, 1)));
                joints[1].SetMotorSpeed(SPEED_KNEE * Neuro.Tools.Sign(action[1]));
                joints[1].SetMotorSpeed((float)(MOTORS_TORQUE * Neuro.Tools.Clip(System.Math.Abs(action[1]), 0, 1)));
                joints[2].SetMotorSpeed(SPEED_HIP * Neuro.Tools.Sign(action[2]));
                joints[2].SetMotorSpeed((float)(MOTORS_TORQUE * Neuro.Tools.Clip(System.Math.Abs(action[2]), 0, 1)));
                joints[3].SetMotorSpeed(SPEED_KNEE * Neuro.Tools.Sign(action[3]));
                joints[3].SetMotorSpeed((float)(MOTORS_TORQUE * Neuro.Tools.Clip(System.Math.Abs(action[3]), 0, 1)));
            }

            World.Step(1.0f / FPS, 6 * 30, 2 * 30);

            var pos = hull.GetPosition();
            var vel = hull.GetLinearVelocity();

            for (int i = 0; i < 10; ++i)
            {
                lidar[i].Fraction = 1.0f;
                lidar[i].P1 = pos;
                lidar[i].P2 = new b2Vec2(pos.x + (float)System.Math.Sin(1.5 * i / 10.0) * LIDAR_RANGE,
                                          pos.y - (float)System.Math.Cos(1.5 * i / 10.0) * LIDAR_RANGE);

                World.RayCast(lidar[i], lidar[i].P1, lidar[i].P2);
            }

            var tmp = new []
            {
                hull.GetAngle(), // Normal angles up to 0.5 here, but sure more is possible.
                2.0 * hull.GetAngularVelocity() / FPS,
                0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS, // Normalized to get -1..1 range
                0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                joints[0].GetJointAngle(), // This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
                joints[0].GetJointSpeed() / SPEED_HIP,
                joints[1].GetJointAngle() + 1.0,
                joints[1].GetJointSpeed() / SPEED_KNEE,
                (legs[1].GetUserData() as CustomBodyData).GroundContact ? 1.0 : 0.0,
                joints[2].GetJointAngle(),
                joints[2].GetJointSpeed() / SPEED_HIP,
                joints[3].GetJointAngle() + 1.0,
                joints[3].GetJointSpeed() / SPEED_KNEE,
                (legs[3].GetUserData() as CustomBodyData).GroundContact ? 1.0 : 0.0
            };

            var stateVal = tmp.Concat(lidar.Select(l => l.Fraction).ToArray()).ToArray();
            Debug.Assert(stateVal.Length == 24);
            State = new Tensor(stateVal, State.Shape);

            scroll = pos.x - (VIEWPORT_W / SCALE / 5);

            float shaping = 130 * pos.x / SCALE;   // moving forward is a way to receive reward (normalized to get 300 on completion)
            shaping -= 5.0f * (float)System.Math.Abs(State.GetFlat(0));  // keep head straight, other than that and falling, any behavior is unpunished

            reward = 0;
            if (!float.IsNaN(prev_shaping))
                reward = shaping - prev_shaping;
            prev_shaping = shaping;

            foreach (var a in action.GetValues())
                reward -= 0.00035 * MOTORS_TORQUE * Neuro.Tools.Clip(System.Math.Abs(a), 0, 1);
            // normalized to about -50.0 using heuristic, more optimal agent should spend less

            bool done = false;
            if (game_over || pos.x < 0)
            {
                reward = -100;
                done = true;
            }

            if (pos.x > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP)
                done = true;

            observation = GetObservation();

            return done;
        }

        public override void Dispose()
        {
            Viewer.Dispose();
            Viewer = null;

            base.Dispose();
        }

        private void GenerateTerrain(bool hardcore)
        {
            const int GRASS = 0, STUMP = 1, STAIRS = 2, PIT = 3, _STATES_ = 4;

            int state = GRASS;
            float velocity = 0.0f;
            float y = TERRAIN_HEIGHT;
            int counter = TERRAIN_STARTPAD;
            bool oneshot = false;
            terrain.Clear();
            terrain_x.Clear();
            terrain_y.Clear();

            float original_y = 0;
            float stair_height = 0;
            int stair_width = 0;
            int stair_steps = 0;

            for (int i = 0; i < TERRAIN_LENGTH; ++i)
            {
                float x = i * TERRAIN_STEP;
                terrain_x.Add(x);

                if (state == GRASS && !oneshot)
                {
                    var y_diff = TERRAIN_HEIGHT - y;
                    velocity = 0.8f * velocity + 0.01f * Neuro.Tools.Sign(y_diff);
                    if (i > TERRAIN_STARTPAD)
                        velocity += (float)Rng.NextDouble(-1, 1) / SCALE;   //1
                    y += velocity;
                }
                else if (state == PIT && oneshot)
                {
                    counter = Rng.Next(3, 5);
                    var poly = new [] 
                    {
                        new b2Vec2(x, y),
                        new b2Vec2(x + TERRAIN_STEP, y),
                        new b2Vec2(x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                        new b2Vec2(x, y - 4 * TERRAIN_STEP)
                    };
                    (fd_polygon.shape as b2PolygonShape).Set(poly);
                    
                    var t = CreateStaticBody(fd_polygon);

                    t.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(1, 1, 1), Color2 = new b2Vec3(0.6f, 0.6f, 0.6f) });
                    terrain.Add(t);

                    (fd_polygon.shape as b2PolygonShape).Set(poly.Select(p => new b2Vec2(p.x + TERRAIN_STEP * counter, p.y)).ToArray());

                    t = CreateStaticBody(fd_polygon);

                    t.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(1, 1, 1), Color2 = new b2Vec3(0.6f, 0.6f, 0.6f) });
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
                else if (state== STUMP && oneshot)
                {
                    counter = Rng.Next(1, 3);
                    var poly = new[] 
                    {
                        new b2Vec2(x, y),
                        new b2Vec2(x + counter * TERRAIN_STEP, y),
                        new b2Vec2(x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                        new b2Vec2(x, y + counter * TERRAIN_STEP)
                    };
                    (fd_polygon.shape as b2PolygonShape).Set(poly);
                    var t = CreateStaticBody(fd_polygon);
                    t.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(1, 1, 1), Color2 = new b2Vec3(0.6f, 0.6f, 0.6f) });
                    terrain.Add(t);
                }
                else if (state== STAIRS && oneshot)
                {
                    stair_height = Rng.NextDouble() > 0.5 ? 1 : -1;
                    stair_width = Rng.Next(4, 5);
                    stair_steps = Rng.Next(3, 5);
                    original_y = y;
                    for (int s = 0; s < stair_steps; ++s)
                    {
                        var poly = new[]
                        {
                            new b2Vec2(x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            new b2Vec2(x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                            new b2Vec2(x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                            new b2Vec2(x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                        };
                        (fd_polygon.shape as b2PolygonShape).Set(poly);
                        var t = CreateStaticBody(fd_polygon);
                        t.SetUserData(new CustomBodyData() { Color1 = new b2Vec3(1, 1, 1), Color2 = new b2Vec3(0.6f, 0.6f, 0.6f) });
                        terrain.Add(t);
                    }

                    counter = stair_steps * stair_width;
                }
                else if (state == STAIRS && !oneshot)
                {
                    var s = stair_steps * stair_width - counter - stair_height;
                    var n = s / stair_width;
                    y = original_y + (n * stair_height) * TERRAIN_STEP;
                }

                oneshot = false;
                terrain_y.Add(y);
                counter -= 1;
                if (counter == 0)
                {
                    counter = Rng.Next(TERRAIN_GRASS / 2, TERRAIN_GRASS);
                    if (state == GRASS && hardcore)
                    {
                        state = Rng.Next(1, _STATES_);
                        oneshot = true;
                    }
                    else
                    {
                        state = GRASS;
                        oneshot = true;
                    }
                }
            }
            
            for (int i = 0; i < TERRAIN_LENGTH - 1; ++i)
            {
                var poly = new[]
                {
                    new b2Vec2(terrain_x[i], terrain_y[i]),
                    new b2Vec2(terrain_x[i + 1], terrain_y[i + 1])
                };
                (fd_edge.shape as b2EdgeShape).m_vertex1 = poly[0];
                (fd_edge.shape as b2EdgeShape).m_vertex2 = poly[1];
                var t = CreateStaticBody(fd_edge);
                var color = new b2Vec3(0.3f, i % 2 == 0 ? 1.0f : 0.8f, 0.3f);
                t.SetUserData(new CustomBodyData() { Color1 = color, Color2 = color });
                terrain.Add(t);
                color = new b2Vec3(0.4f, 0.6f, 0.3f);
                var poly_extended = new List<double[]>
                {
                    new double[] { poly[0].x, poly[0].y },
                    new double[] { poly[1].x, poly[1].y },
                    new double[] { poly[1].x, 0 },
                    new double[] { poly[0].x, 0 }
                };
                terrain_poly.Add(new Tuple<List<double[]>, b2Vec3> (poly_extended, color));
            }

            terrain.Reverse();
        }

        private void Destroy()
        {
            foreach (var t in terrain)
                World.DestroyBody(t);

            terrain.Clear();

            if (hull != null)
                World.DestroyBody(hull);
            hull = null;

            foreach (var leg in legs)
                World.DestroyBody(leg);

            legs.Clear();
            joints.Clear();
        }

        b2Body CreateStaticBody(b2FixtureDef b2FixtureDef)
        {
            b2BodyDef b = new b2BodyDef();
            b.type = 0;
            var body = World.CreateBody(b);
            body.CreateFixture(b2FixtureDef);
            return body;
        }

        b2Body CreateDynamicBody(b2Vec2 position, float angle, b2FixtureDef b2FixtureDef)
        {
            var body = World.CreateBody(new b2BodyDef());
            body.CreateFixture(b2FixtureDef);
            body.SetTransform(position, angle);
            return body;
        }

        private const int FPS = 50;
        private const float SCALE = 30.0f;   // affects how fast-paced the game is, forces should be adjusted as well

        private const float MOTORS_TORQUE = 80;
        private const float SPEED_HIP = 4;
        private const float SPEED_KNEE = 6;
        private const float LIDAR_RANGE = 160 / SCALE;

        private const float INITIAL_RANDOM = 5;

        private readonly List<float[]> HULL_POLY = new List<float[]> { new[] { -30.0f, +9 }, new[] { +6.0f, +9 }, new[] { +34.0f, +1 }, new[] { +34.0f, -8 }, new[] { -30.0f, -8 } };

        private const float LEG_DOWN = -8 / SCALE;
        private const float LEG_W = 8 / SCALE;
        private const float LEG_H = 34 / SCALE;

        private const int VIEWPORT_W = 600;
        private const int VIEWPORT_H = 400;

        private const float TERRAIN_STEP = 14 / SCALE;
        private const int TERRAIN_LENGTH = 200;     // in steps
        private const float TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4;
        private const int TERRAIN_GRASS = 10;    // low long are grass spots, in steps
        private const int TERRAIN_STARTPAD = 20;    // in steps
        private const float FRICTION = 2.5f;

        private b2FixtureDef HULL_FD = new b2FixtureDef();
        private b2FixtureDef LEG_FD = new b2FixtureDef();
        private b2FixtureDef LOWER_FD = new b2FixtureDef();

        private Rendering.Viewer Viewer;
        private b2World World;
        private b2FixtureDef fd_polygon = new b2FixtureDef();
        private b2FixtureDef fd_edge = new b2FixtureDef();
        private List<b2Body> terrain = new List<b2Body>();
        private List<float> terrain_x = new List<float>();
        private List<float> terrain_y = new List<float>();
        private List<b2Body> legs = new List<b2Body>();
        private List<b2RevoluteJoint> joints = new List<b2RevoluteJoint>();
        private b2Body hull;
        private List<Tuple<List<double[]>, b2Vec3>> terrain_poly = new List<Tuple<List<double[]>, b2Vec3>>();
        private List<b2Body> drawlist = new List<b2Body>();
        private bool hardcore = false;
        private bool game_over;
        private float scroll;
        private float prev_shaping;
        private int lidar_render;
        private LidarCallback[] lidar = new LidarCallback[10];

        private class ContactDetector : b2ContactListener
        {
            public ContactDetector(BipedalWalkerEnv env)
            {
                Env = env;
            }

            public override void BeginContact(b2Contact contact)
            {
                if (Env.hull == contact.GetFixtureA().GetBody() || Env.hull == contact.GetFixtureB().GetBody())
                    Env.game_over = true;
                foreach (var leg in new[] { Env.legs[1], Env.legs[3] })
                    if (leg == contact.GetFixtureA().GetBody() || leg == contact.GetFixtureB().GetBody())
                        (leg.GetUserData() as CustomBodyData).GroundContact = true;
            }

            public override void EndContact(b2Contact contact)
            {
                foreach (var leg in new[] { Env.legs[1], Env.legs[3] })
                    if (leg == contact.GetFixtureA().GetBody() || leg == contact.GetFixtureB().GetBody())
                        (leg.GetUserData() as CustomBodyData).GroundContact = false;
            }

            private BipedalWalkerEnv Env;
        }

        private ContactDetector contactDetector;

        private class CustomBodyData
        {
            public b2Vec3 Color1;
            public b2Vec3 Color2;
            public bool GroundContact;
        }
    }
}
