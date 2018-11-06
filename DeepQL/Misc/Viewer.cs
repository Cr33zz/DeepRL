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
        // Implementation based upon https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
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
                OpenGLControl.OpenGL.BlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
                OpenGLControl.OpenGL.Viewport(0, 0, Width, Height);
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
                gl.ClearColor(0.4f, 0.45f, 0.5f, 1.0f);

                gl.MatrixMode(OpenGL.GL_PROJECTION);
                gl.LoadIdentity();
                gl.Ortho(0, Width, 0, Height, - 10, 10);
                gl.MatrixMode(OpenGL.GL_MODELVIEW);
                gl.LoadIdentity();

                gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);

                Trans.Enable(gl);
                foreach (var geom in Geoms)
                    geom.Render(gl);
                foreach (var geom in OneTimeGeoms)
                    geom.Render(gl);
                Trans.Disable(gl);

                OneTimeGeoms.Clear();
            }

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
                gl.Translate(Translation[0], Translation[1], 0);
                gl.Rotate(RAD2DEG * Rotation, 0, 0, 1.0);
                gl.Scale(Scale[0], Scale[1], 1);
            }

            public override void Disable(OpenGL gl)
            {
                gl.PopMatrix();
            }

            public void SetTranslation(double newX, double newY)
            {
                Translation[0] = newX;
                Translation[1] = newY;
            }

            public void SetRotation(double rot)
            {
                Rotation = rot;
            }

            public void SetScale(double newX, double newY)
            {
                Scale[0] = newX;
                Scale[1] = newY;
            }

            private double[] Translation = new double[2];
            private double Rotation;
            private double[] Scale = new double[2] {1, 1};

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
            public FilledPolygon(List<double[]> vertices)
            {
                Vertices = new List<double[]>(vertices);
            }

            protected override void OnRender(OpenGL gl)
            {
                if (Vertices.Count == 4)
                    gl.Begin(OpenGL.GL_QUADS);
                else if (Vertices.Count > 4)
                    gl.Begin(OpenGL.GL_POLYGON);
                else
                    gl.Begin(OpenGL.GL_TRIANGLES);

                foreach (var p in Vertices)
                    gl.Vertex(p[0], p[1], 0); // draw each vertex

                gl.End();
            }

            public List<double[]> Vertices;
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
            public PolyLine(List<double[]> vertices, bool close)
            {
                Vertices = new List<double[]>(vertices);
                Close = close;
            }

            protected override void OnRender(OpenGL gl)
            {
                gl.Begin(Close ? OpenGL.GL_LINE_LOOP : OpenGL.GL_LINE_STRIP);
                foreach (var p in Vertices)
                    gl.Vertex(p[0], p[1], 0); // draw each vertex
                gl.End();
            }

            public List<double[]> Vertices;
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
    }
}
