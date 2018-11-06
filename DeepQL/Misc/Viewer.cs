using SharpGL;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace DeepQL.Misc
{
    public static class Rendering
    {
        public class Viewer : Form
        {
            public Viewer(int width, int height)
            {
                ClientSize = new Size(width, height);
                ((System.ComponentModel.ISupportInitialize)(OpenGLControl)).BeginInit();
                OpenGLControl.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
                OpenGLControl.DrawFPS = false;
                OpenGLControl.FrameRate = 28;
                OpenGLControl.Location = new System.Drawing.Point(0, 0);
                OpenGLControl.Name = "OpenGL";
                OpenGLControl.RenderContextType = SharpGL.RenderContextType.FBO;
                OpenGLControl.RenderTrigger = RenderTrigger.Manual;
                OpenGLControl.Size = new System.Drawing.Size(width, height);
                OpenGLControl.TabIndex = 0;
                OpenGLControl.OpenGLDraw += new RenderEventHandler(OpenGLDrawFunc);
                OpenGLControl.OpenGL.Enable(OpenGL.GL_BLEND);
                OpenGLControl.OpenGL.Enable(OpenGL.GL_CULL_FACE);
                OpenGLControl.OpenGL.BlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
                OpenGLControl.OpenGL.Disable(OpenGL.GL_DEPTH_TEST);
                OpenGLControl.OpenGL.Disable(OpenGL.GL_SCISSOR_TEST);
                OpenGLControl.OpenGL.Viewport(0, 0, width, height);
                Controls.Add(OpenGLControl);
                ((System.ComponentModel.ISupportInitialize)(OpenGLControl)).EndInit();

                Name = Text = "Viewer";
                FormBorderStyle = FormBorderStyle.None;

                Show();
            }

            //public void SetBounds(int left, int right, int bottom, int top)
            //{
            //    double scaleX = (double)Width / (right - left);
            //    double scaleY = (double)Height / (top - bottom);

            //    Trans.SetTranslation(-left * scaleX, -bottom * scaleY);
            //    Trans.SetScale(scaleX, scaleY);
            //}

            public void AddGeom(Geom geom)
            {
                Geoms.Add(geom);
            }

            public void AddOneTime(Geom geom)
            {
                OneTimeGeoms.Add(geom);
            }

            public void ManualRender()
            {
                OpenGLControl.DoRender();
                Application.DoEvents();
            }

            private void OpenGLDrawFunc(object sender, RenderEventArgs e)
            {
                OpenGL gl = OpenGLControl.OpenGL;

                gl.ClearColor(0.2f, 0.25f, 0.3f, 1.0f);
                gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT/* | OpenGL.GL_DEPTH_BUFFER_BIT*/);

                int[] viewport = new int[4];
                gl.Ortho(0, Width, 0, Height, -1, 1);

                ////  Create the appropriate modelview matrix.
                //gl.MatrixMode(OpenGL.GL_MODELVIEW);
                //gl.PushMatrix();
                //gl.LoadIdentity();



                //gl.Translate(0, 0, 0);               // Move Left And Into The Screen
                //gl.Rotate(rtri, 0.0f, 1.0f, 0.0f);				// Rotate The Pyramid On It's Y Axis
                gl.Begin(OpenGL.GL_TRIANGLES);                  // Start Drawing The Pyramid
                gl.Color(1.0f, 0.0f, 0.0f);         // Red
                gl.Vertex(50, 0);            // Top Of Triangle (Front)
                gl.Color(0.0f, 1.0f, 0.0f);         // Green
                gl.Vertex(0, 100);          // Left Of Triangle (Front)
                gl.Color(0.0f, 0.0f, 1.0f);         // Blue
                gl.Vertex(100, 100);           // Right Of Triangle (Front)
                gl.End();						// Done Drawing The Pyramid




                //Trans.Enable(gl);
                //foreach (var geom in Geoms)
                //    geom.Render(gl);
                //foreach (var geom in OneTimeGeoms)
                //    geom.Render(gl);
                //Trans.Disable(gl);

                //OneTimeGeoms.Clear();

                //gl.MatrixMode(OpenGL.GL_MODELVIEW);
                //gl.Disable(OpenGL.GL_DEPTH_TEST);

                rtri += 3.0f;// 0.2f;						// Increase The Rotation Variable For The Triangle 
            }

            float rtri = 0;


            protected override void Dispose(bool disposing)
            {
                OpenGLControl.Dispose();
                base.Dispose(disposing);
            }

            private readonly OpenGLControl OpenGLControl = new OpenGLControl();
            private readonly List<Geom> Geoms = new List<Geom>();
            private readonly List<Geom> OneTimeGeoms = new List<Geom>();
            private readonly Transform Trans = new Transform();
        }

        public abstract class Geom
        {
            protected Geom()
            {
                _Color = new Color(new double[] {0, 0, 0, 1});
                AddAttr(_Color);
            }

            public void Render(OpenGL gl)
            {
                for (int i = Attrs.Count - 1; i >= 0; --i)
                    Attrs[i].Enable(gl);
                OnRender(gl);
                for (int i = Attrs.Count - 1; i >= 0; --i)
                    Attrs[i].Disable(gl);
            }

            protected abstract void OnRender(OpenGL gl);

            public void AddAttr(Attr attr)
            {
                Attrs.Add(attr);
            }

            public void SetColor(double r, double g, double b)
            {
                _Color.Vec4 = new double[] {r, g, b, 1};
            }

            private Color _Color;
            private readonly List<Attr> Attrs = new List<Attr>();
        }

        public abstract class Attr
        {
            public abstract void Enable(OpenGL gl);
            public virtual void Disable(OpenGL gl) { }
        }

        public class Transform : Attr
        {
            public Transform(double[] translation = null, double rotation = 0, double[] scale = null)
            {
                if (translation != null) SetTranslation(translation[0], translation[1]);
                SetRotation(rotation);
                if (scale != null) SetScale(scale[0], scale[1]);
            }

            public override void Enable(OpenGL gl)
            {
                gl.PushMatrix();
                gl.Translate(translation[0], translation[1], 0);
                gl.Rotate(RAD2DEG * rotation, 0, 0, 1.0);
                gl.Scale(scale[0], scale[1], 1);
            }

            public override void Disable(OpenGL gl)
            {
                gl.PopMatrix();
            }

            public void SetTranslation(double newX, double newY)
            {
                translation[0] = newX;
                translation[1] = newY;
            }

            public void SetRotation(double rot)
            {
                rotation = rot;
            }

            public void SetScale(double newX, double newY)
            {
                scale[0] = newX;
                scale[1] = newY;
            }

            private double[] translation = new double[2];
            private double rotation;
            private double[] scale = new double[2] {1, 1};

            private const double RAD2DEG = 57.29577951308232;
        }

        public class Color : Attr
        {
            public Color(double[] vec4)
            {
                Vec4 = (double[]) vec4.Clone();
            }

            public override void Enable(OpenGL gl)
            {
                gl.Color(Vec4);
            }

            public double[] Vec4;
        }

        public class Point : Geom
        {
            protected override void OnRender(OpenGL gl)
            {
                gl.Begin(OpenGL.GL_POINTS);
                gl.Vertex(0.0, 0.0, 0.0);
                gl.End();
            }
        }

        public class FilledPolygon : Geom
        {
            public FilledPolygon(List<double[]> v)
            {
                V = new List<double[]>(v);
            }

            protected override void OnRender(OpenGL gl)
            {
                if (V.Count == 4)
                    gl.Begin(OpenGL.GL_QUADS);
                else if (V.Count > 4)
                    gl.Begin(OpenGL.GL_POLYGON);
                else
                    gl.Begin(OpenGL.GL_TRIANGLES);

                foreach (var p in V)
                    gl.Vertex(p[0], p[1], 0); // draw each vertex

                gl.End();
            }

            public List<double[]> V;
        }

        public class Line : Geom
        {
            public Line(double[] start, double[] end)
            {
                Start = (double[]) start.Clone();
                End = (double[]) end.Clone();
            }

            protected override void OnRender(OpenGL gl)
            {
                //gl.LineWidth(2);
                gl.Begin(OpenGL.GL_LINES);
                gl.Vertex(Start[0], Start[1], 0);
                gl.Vertex(End[0], End[1], 0);
                gl.End();
            }

            public double[] Start;
            public double[] End;
        }

        public class PolyLine : Geom
        {
            public PolyLine(List<double[]> v, bool close)
            {
                V = new List<double[]>(v);
                Close = close;
            }

            protected override void OnRender(OpenGL gl)
            {
                gl.Begin(Close ? OpenGL.GL_LINE_LOOP : OpenGL.GL_LINE_STRIP);
                foreach (var p in V)
                    gl.Vertex(p[0], p[1], 0); // draw each vertex
                gl.End();
            }

            public List<double[]> V;
            public bool Close;
        }

        public class Compound : Geom
        {
            public Compound(Geom[] geoms)
            {
                Geoms = geoms;
            }

            protected override void OnRender(OpenGL gl)
            {
                foreach (var geom in Geoms)
                    geom.Render(gl);
            }

            private readonly Geom[] Geoms;
        }

        public static Geom MakeCircle(double radius = 10, int res = 30, bool filled = true)
        {
            List<double[]> points = new List<double[]>();
            for (int i = 0; i < res; ++i)
            {
                double ang = 2 * Math.PI * i / res;
                points.Add(new[] {Math.Cos(ang) * radius, Math.Sin(ang) * radius});
            }

            if (filled)
                return new FilledPolygon(points);
            return new PolyLine(points, true);
        }

        public static Geom MakePolygon(List<double[]> v, bool filled = true)
        {
            if (filled)
                return new FilledPolygon(v);
            return new PolyLine(v, true);
        }

        public static Geom MakePolyLine(List<double[]> v)
        {
            return new PolyLine(v, false);
        }

        public static Geom MakeCapsule(double length, double width)
        {
            double l = length, r = width / 2, t = -width / 2, b = 0;
            var box = MakePolygon(new List<double[]> {new[] {l, b}, new[] {l, t}, new[] {r, t}, new[] {r, b}});
            var circ0 = MakeCircle(width / 2);
            var circ1 = MakeCircle(width / 2);
            circ1.AddAttr(new Transform(new[] {length, 0}));
            return new Compound(new[] {box, circ0, circ1});
        }

        //https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
    }
}
