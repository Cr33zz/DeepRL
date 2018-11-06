using System;
using System.Collections.Generic;
using System.Linq;
using DeepQL;
using DeepQL.Environments;
using DeepQL.Spaces;
using Neuro.Tensors;
using Point = System.Tuple<int, int>;

namespace DeepQL.Gyms
{
    /**
    The Taxi Problem (implementation based on OpenAI gym https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations. 
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
    **/
    public class TaxiEnv : DiscreteEnv
    {
        public TaxiEnv()
            : base(STATES_NUM, ACTIONS_NUM)
        {
            const int rowsNum = 5;
            const int colsNum = 5;
            const int maxRow = rowsNum - 1;
            const int maxCol = colsNum - 1;

            for (int row = 0; row < 5; ++row)
            for (int col = 0; col < 5; ++col)
            for (int passIdx = 0; passIdx < 5; ++passIdx)
            for (int destidx = 0; destidx < 4; ++destidx)
            {
                int state = EncodeState(row, col, passIdx, destidx);
                if (passIdx < 4 && passIdx != destidx)
                    SetInitialStateDistribution(state, 1);

                for (int a = 0; a < ACTIONS_NUM; ++a)
                {
                    //defaults
                    int newRow = row;
                    int newCol = col;
                    int newPassIdx = passIdx;

                    double reward = -1;
                    bool done = false;
                    Point taxiLoc = new Point(row, col);

                    if (a == 0)
                        newRow = Math.Min(row + 1, maxRow);
                    else if (a == 1)
                        newRow = Math.Max(row - 1, 0);

                    if (a == 2 && MAP[1 + row][2 * col + 2] == ':')
                    {
                        newCol = Math.Min(col + 1, maxCol);
                    }
                    else if (a == 3 && MAP[1 + row][2 * col] == ':')
                    {
                        newCol = Math.Max(col - 1, 0);
                    }
                    else if (a == 4) // pickup
                    {
                        if (passIdx < 4 && taxiLoc.Equals(LOCATIONS_INDICES[passIdx]))
                            newPassIdx = 4;
                        else
                            reward = -10;
                    }
                    else if (a == 5) // dropoff
                    {
                        if (taxiLoc.Equals(LOCATIONS_INDICES[destidx]) && passIdx == 4)
                        {
                            done = true;
                            reward = 20;
                        }
                        else if (LOCATIONS_INDICES.Contains(taxiLoc) && passIdx == 4)
                        {
                            newPassIdx = LOCATIONS_INDICES.IndexOf(taxiLoc);
                        }
                        else
                        {
                            reward = -10;
                        }
                    }

                    int newState = EncodeState(newRow, newCol, newPassIdx, destidx);
                    AddTransition(state, a, 1.0, newState, reward, done);
                }
            }

            FinalizeInitialStateDistribution();
        }

        public override byte[] Render(bool toRgbArray = false)
        {
            OutputColorizer colorizer = OutputColorizer.FromStringsAsChars(MAP);

            int taxiRow, taxiCol, passIdx, destIdx;
            DecodeState(StateAsInt, out taxiRow, out taxiCol, out passIdx, out destIdx);

            if (passIdx < 4)
            {
                int pi = LOCATIONS_INDICES[passIdx].Item1, pj = LOCATIONS_INDICES[passIdx].Item2;
                colorizer.Override(1 + pi, 2 * pj + 1, MAP[1 + pi][2 * pj + 1].ToString(), ConsoleColor.Blue);
            }

            int di = LOCATIONS_INDICES[destIdx].Item1, dj = LOCATIONS_INDICES[destIdx].Item2;
            colorizer.Override(1 + di, 2 * dj + 1, MAP[1 + di][2 * dj + 1].ToString(), ConsoleColor.Magenta);

            colorizer.Override(1 + taxiRow, 2 * taxiCol + 1, MAP[1 + taxiRow][2 * taxiCol + 1].ToString(), passIdx < 4 ? ConsoleColor.Yellow : ConsoleColor.Green, true);

            var actionName = new [] {"South", "North", "East", "West", "Pickup", "Dropoff"};
            if (LastActionAsInt >= 0)
                colorizer.AddLine(actionName[LastActionAsInt]);

            Console.Clear();
            colorizer.Print();

            return null;
        }

        private int EncodeState(int taxiRow, int taxiCol, int passIdx, int destIdx)
        {
            int state = taxiRow;
            state *= 5;
            state += taxiCol;
            state *= 5;
            state += passIdx;
            state *= 4;
            state += destIdx;
            return state;
        }

        private void DecodeState(int state, out int taxiRow, out int taxiCol, out int passIdx, out int destIdx)
        {
            destIdx = state % 4;
            state = state / 4;
            passIdx = state % 5;
            state = state / 5;
            taxiCol = state % 5;
            state = state / 5;
            taxiRow = state;
        }

        private readonly string[] MAP = 
        {
            "+---------+",
            "|R: | : :G|",
            "| : : : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        };

        private const int ACTIONS_NUM = 6;
        private const int STATES_NUM = 5*5*5*4;
        private readonly List<Point> LOCATIONS_INDICES = new List<Point> { new Point(0, 0), new Point(0, 4), new Point(4, 0), new Point(4, 3) };        
    }
}
