using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Forms;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.Misc;
using Neuro.Tensors;

namespace Examples
{
    class CartPole
    {
        static void Main(string[] args)
        {
            Env env = new CartPoleEnv();

            env.Reset();

            Rendering.Viewer v = new Rendering.Viewer(600, 400);
            //var poly = Rendering.MakePolygon(new List<double[]>() { new[] { -10, -10.0 }, new[] { -10, 10.0 }, new[] { 10, 10.0 }, new[] { 10, -10.0 } });
            //poly.SetColor(1, 0, 0);
            //v.AddGeom(poly);
            var track = new Rendering.Line(new[] { 0, 100.0 }, new[] { 600, 100.0 });
            track.SetColor(0, 0, 0);
            v.AddGeom(track);

            //while (!env.Step((int)env.ActionSpace.Sample()[0], out var nextState, out var reward))
            while (true)
            {
                //env.Render();
                v.ManualRender();
                Thread.Sleep(30);
            }
                

            return;
        }
    }
}
