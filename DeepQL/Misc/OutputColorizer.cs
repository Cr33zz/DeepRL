using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    // Naive implementation of class allowing printing colored output to console output
    public class OutputColorizer
    {
        public static OutputColorizer FromStringsAsChars(IEnumerable<string> strings)
        {
            var colorizer = new OutputColorizer();
            foreach (var line in strings)
            {
                foreach (var c in line)
                {
                    colorizer.Add(c.ToString());
                }
                colorizer.AddLine("");
            }
            return colorizer;
        }

        public static OutputColorizer FromStrings(IEnumerable<string> strings)
        {
            var colorizer = new OutputColorizer();
            foreach (var line in strings)
                colorizer.AddLine(line);
            return colorizer;
        }

        public void AddLine(string s, ConsoleColor color = ConsoleColor.White, bool highlight = false)
        {
            Add(s, color, highlight, true);
        }

        public void Add(string s, ConsoleColor color = ConsoleColor.White, bool highlight = false, bool newLine = false)
        {
            if (Lines.Count == 0)
                Lines.Add(new List<ColoredStr>());

            if (s.Length > 0)
                Lines.Last().Add(new ColoredStr() { Str = s, Color = color, Highlight = highlight });

            if (newLine)
                Lines.Add(new List<ColoredStr>());
        }

        public void Override(int line, int col, string s, ConsoleColor color = ConsoleColor.White, bool highlight = false)
        {
            Lines[line][col] = new ColoredStr() { Str = s, Color = color, Highlight = highlight };
        }

        public void Print()
        {
            Console.ResetColor();

            foreach (var line in Lines)
            {
                foreach (var coloredStr in line)
                {
                    if (coloredStr.Highlight)
                        Console.BackgroundColor = coloredStr.Color;
                    else
                        Console.ForegroundColor = coloredStr.Color;

                    Console.Write(coloredStr.Str);
                    Console.ResetColor();
                }
                Console.WriteLine();
            }
        }

        private class ColoredStr
        {
            public string Str;
            public ConsoleColor Color;
            public bool Highlight;
        }

        private readonly List<List<ColoredStr>> Lines = new List<List<ColoredStr>>();
    }
}
