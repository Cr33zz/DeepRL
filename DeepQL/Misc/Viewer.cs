using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace DeepQL.Misc
{
    public class Viewer : Form
    {
        public Viewer(int width, int height)
        {
            Name = Text = "Viewer";
            DoubleBuffered = true;
            Paint += new PaintEventHandler(Render);

            update_timer.Enabled = true;
            update_timer.Interval = 5;
            update_timer.Tick += new EventHandler(Update);
        }

        protected virtual void Render(object sender, PaintEventArgs e)
        {
            e.Graphics.CompositingQuality = CompositingQuality.GammaCorrected;
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        }

        protected virtual void Update(object sender, EventArgs e)
        {
            Refresh();
        }

        private Timer update_timer = new Timer();

        public static readonly Font SMALL_FONT = new Font("Arial", 8);
        public static readonly Font SMALL_BOLD_FONT = new Font("Arial", 8, FontStyle.Bold);
        public static readonly Font NORMAL_FONT = new Font("Arial", 9);
        public static readonly Font NORMAL_BOLD_FONT = new Font("Arial", 9, FontStyle.Bold);
        public static readonly Font BIG_FONT = new Font("Arial", 10);
        public static readonly Font BIG_BOLD_FONT = new Font("Arial", 10, FontStyle.Bold);

        public static readonly Pen BOLD_BLACK_PEN = new Pen(Color.Black, 3);
        public static readonly Pen BOLD_WHITE_PEN = new Pen(Color.White, 3);
        public static readonly Pen BOLD_RED_PEN = new Pen(Color.Red, 3);
        public static readonly Pen BOLD_GREEN_PEN = new Pen(Color.Green, 3);
        public static readonly Pen BOLD_BLUE_PEN = new Pen(Color.Blue, 3);
        public static readonly Pen BOLD_LIGHT_BLUE_PEN = new Pen(Color.LightBlue, 3);
    }

    public abstract class Geom
    {
        public Geom()
        {
            Color = Color.Black;
        }

        public void Render()
        {
            //for (var attr in Attrs.reverse)
            //    attr.Enable();
            //OnRender();
            //for (var attr in Attrs.reverse)
            //    attr.Disable();
        }

        public abstract void OnRender();

        public void AddAttr(Attr attr)
        {
            Attrs.Add(attr);
        }

        public void SetColor(int r, int g, int b)
        {
            Color = Color.FromArgb(r, g, b);
        }

        private Color Color;
        private List<Attr> Attrs = new List<Attr>();
    }

    public abstract class Attr
    {
        public abstract void Enable();
        public virtual void Disable() { }
    }

    public class Transform : Attr
    {
        public Transform(double[] translation = null, double rotation = 0, double[] scale = null)
        {
            if (translation != null) SetTranslation(translation[0], translation[1]);
            SetRotation(rotation);
            if (scale != null) SetScale(scale[0], scale[1]);
        }
        public override void Enable()
        {
            //glPushMatrix()
            //glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
            //glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
            //glScalef(self.scale[0], self.scale[1], 1)
        }
        public override void Disable()
        {
            //glPopMatrix()
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
        private double[] scale = new double[2] { 1, 1 };
    }

    public class Color : Attr
    {
        public Color(float[] vec4)
        {
            Vec4 = Array.Copy(vec4, Vec4, 4);
        }

        public override void Enable()
        {
            //glColor4f(*self.vec4)
        }

        private float[] Vec4 = new float[4];
    }

    //https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
}
