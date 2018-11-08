using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using AleControlLib;

namespace AleSample
{
    /// <summary>
    /// This sample is provided to show how to initialize and use the AleControl.  However, before
    /// using this sample, you should complete the configuration steps noted in the remarks.
    /// </summary>
    /// <remarks>
    /// This sample expects that the following configuration steps have been completed.
    /// 
    /// 1.) Register the file 'packages\AleControl.x.x.x.x\nativeBinaries\x64\AleControl.dll'
    ///     by running the command 'regsvr32.exe AleControl.dll' from a CMD window with
    ///     Administrative privileges.
    /// 2.) IMPORTANT: Setup your project to build to the x64 Platform Target.
    /// </remarks>
    public class AleSample
    {
        public AleSample()
        {
        }

        /// <summary>
        /// This static function shows how to use the AleControl in with a very simple example.
        /// </summary>
        /// <remarks>
        /// The following sample, creates the ALE envrionment and uses it to run the ATARI game
        /// 'pong'.  
        /// 
        /// To find numerous other ROMS, please see https://github.com/openai/atari-py/tree/master/atari_py/atari_roms
        /// </remarks>
        public static void RunSample()
        {
            string strRomDir = Path.GetFullPath("..\\..\\roms\\");
            string strRom = strRomDir + "pong.bin";
            IALE ale = new ALE();

            ale.Initialize();
            ale.EnableDisplayScreen = true;
            ale.EnableSound = true;
            ale.EnableColorData = true;
            ale.EnableRestrictedActionSet = true;
            ale.EnableColorAveraging = true;
            ale.RandomSeed = 123;
            ale.RepeatActionProbability = 0.25f;
            ale.Load(strRom);
            ACTION[] rgActions = ale.ActionSpace;
            Random rand = new Random();
            int nEpisode = 0;
            float fWid;
            float fHt;
            float fTotalReward = 0;

            ale.GetScreenDimensions(out fWid, out fHt);

            while (nEpisode < 10)
            {
                ACTION action = rgActions[rand.Next(rgActions.Length)];

                fTotalReward += ale.Act(action);

                byte[] rgScreen = ale.GetScreenData(COLORTYPE.CT_COLOR);
                // Do something with the screen data here.

                if (ale.GameOver)
                {
                    ale.ResetGame();
                    nEpisode++;
                    fTotalReward = 0;
                }
            }

            ale.Shutdown();
        }
    }
}
