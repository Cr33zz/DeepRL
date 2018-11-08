using AleControlLib;
using DeepQL.Environments;
using DeepQL.Misc;
using DeepQL.Spaces;
using Neuro.Tensors;
using System;
using System.IO;

namespace DeepQL.Gyms
{
    public class AtariEnv : Env
    {
        public AtariEnv(string gameRomFile, bool grayscaleObs = false)
            : base(null, null)
        {
            string strRom = Path.GetFullPath(gameRomFile);
            ale.Initialize();
            ale.EnableDisplayScreen = false;
            ale.EnableSound = false;
            ale.EnableColorData = true;
            ale.EnableRestrictedActionSet = true;
            ale.EnableColorAveraging = true;
            ale.RandomSeed = 123;
            ale.RepeatActionProbability = 0.25f;
            ale.Load(strRom);

            rgActions = ale.ActionSpace;
            Random rand = new Random();
            
            ale.GetScreenDimensions(out var screenWidth, out var screenHeight);

            ActionSpace = new Discrete(rgActions.Length);
            ObservationSpace = new Box(0, 255, new Shape((int)screenWidth, (int)screenHeight, grayscaleObs ? 1 : 3));
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            if (Viewer == null)
            {
                Viewer = new Rendering.Viewer(ObservationSpace.Shape.Width, ObservationSpace.Shape.Height);
            }

            byte[] rgScreen = ale.GetScreenData(COLORTYPE.CT_COLOR);
            var img = new Rendering.Image(rgScreen, ObservationSpace.Shape.Width, ObservationSpace.Shape.Height);
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
            byte[] rgScreen = ale.GetScreenData(COLORTYPE.CT_COLOR);
            Tensor obs = new Tensor(ObservationSpace.Shape);
            obs.FillWithPixelData(rgScreen);
            return obs;
        }

        public override void Dispose()
        {
            ale.Shutdown();
            base.Dispose();
        }

        private IALE ale = new ALE();
        private ACTION[] rgActions;

        private Rendering.Viewer Viewer;
        private Rendering.Transform CarTrans;
    }
}
