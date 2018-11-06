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

            Rendering.Viewer v = new Rendering.Viewer(800, 600);
            var poly = Rendering.MakePolygon(new List<double[]>() { new[] { 0, 0.0 }, new[] { 100, 0.0 }, new[] { 100, 100.0 }, new[] { 0, 100.0 } });
            poly.SetColor(1,0,0);
            v.AddGeom(poly);
            v.Show();
            
            //while (!env.Step((int)env.ActionSpace.Sample()[0], out var nextState, out var reward))
            while (true)
            {
                v.ManualRender();
                Thread.Sleep(30);
            }
                

            return;
        }
    }
}
