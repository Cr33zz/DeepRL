using AleControlLib;
using DeepQL.Environments;
using DeepQL.Misc;
using DeepQL.Spaces;
using Neuro.Tensors;
using System;
using System.IO;
using System.Linq;

namespace DeepQL.Gyms
{
    public class AtariEnv : Env
    {
        public AtariEnv(string romFile, bool grayscaleObs = false)
            : base(null, null)
        {
            if (!File.Exists(romFile))
                throw new FileNotFoundException($"Provided ROM file '{romFile}' doesn't exist.");

            string strRom = Path.GetFullPath(romFile);
            ale.Initialize();
            ale.EnableDisplayScreen = false;
            ale.EnableSound = false;
            ale.EnableColorData = true;
            ale.EnableRestrictedActionSet = true;
            ale.EnableColorAveraging = true;
            ale.RepeatActionProbability = 0;//0.25f;
            ale.Load(strRom);

            rgActions = ale.ActionSpace;
            Random rand = new Random();
            
            ale.GetScreenDimensions(out var screenWidth, out var screenHeight);

            ActionSpace = new Discrete(rgActions.Length);
            ObservationSpace = new Box(0, 255, new Shape((int)screenWidth, (int)screenHeight, grayscaleObs ? 1 : 3));
        }

        public override void Seed(int seed = 0)
        {
            ale.RandomSeed = seed > 0 ? seed : Rng.Next();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(ObservationSpace.Shape.Width, ObservationSpace.Shape.Height);
            }

            var img = new Rendering.Image(GetScreenPixelData(), ObservationSpace.Shape.Width, ObservationSpace.Shape.Height);
            Viewer.AddOneTime(img);

            Viewer.Render();
            return null;
        }

        public override Tensor Reset()
        {
            ale.ResetGame();
            return GetObservation();
        }

        public override bool Step(Tensor action, out Tensor observation, out double reward)
        {
            ACTION a = rgActions[(int)action[0]];
            reward = ale.Act(a);
            observation = GetObservation();
            return ale.GameOver;
        }

        protected override Tensor GetObservation()
        {
            byte[] screenData = ale.GetScreenData(ObservationSpace.Shape.Depth == 1 ? COLORTYPE.CT_GRAYSCALE : COLORTYPE.CT_COLOR);
            Tensor obs = new Tensor(ObservationSpace.Shape);
            obs.FillWithPixelData(screenData);
            return obs;
        }

        public override void Dispose()
        {
            ale.Shutdown();
            base.Dispose();
        }

        private byte[] GetScreenPixelData()
        {
            bool grayscale = ObservationSpace.Shape.Depth == 1;

            byte[] screenData = ale.GetScreenData(grayscale ? COLORTYPE.CT_GRAYSCALE : COLORTYPE.CT_COLOR);

            int PIXELS_PER_COLOR = ObservationSpace.Shape.Width * ObservationSpace.Shape.Height;

            byte[] pixelData = new byte[PIXELS_PER_COLOR * 3];

            int SCREEN_DATA_OFFSET_ENABLED = grayscale ? 0 : 1;

            // Since the first positions of screenData contain the red colors, followed by the green colors and then the blue colors,
            // we have to convert them to RGB pixels sequence.

            for (int y = 0; y < ObservationSpace.Shape.Height; ++y)
            for (int x = 0; x < ObservationSpace.Shape.Width; ++x)
            {
                int pixelIdx = y * ObservationSpace.Shape.Width + x;
                int dataIdx = (ObservationSpace.Shape.Height - y - 1) * ObservationSpace.Shape.Width + x;

                pixelData[pixelIdx * 3] = (byte)(screenData[dataIdx] * (grayscale ? 2 : 1));
                pixelData[pixelIdx * 3 + 1] = (byte)(screenData[PIXELS_PER_COLOR * SCREEN_DATA_OFFSET_ENABLED + dataIdx] * (grayscale ? 2 : 1));
                pixelData[pixelIdx * 3 + 2] = (byte)(screenData[2 * PIXELS_PER_COLOR * SCREEN_DATA_OFFSET_ENABLED + dataIdx] * (grayscale ? 2 : 1));
            }

            return pixelData;
        }

        private IALE ale = new ALE();
        private ACTION[] rgActions;

        private Rendering.Viewer Viewer;
    }
}
