using System.Linq;
using AutoDiff;
using FluentAssertions;

namespace ML.Utility
{
    /// <summary>
    ///     Todo
    /// </summary>
    public class TermMatrix
    {
        public TermMatrix(int witdh, int height)
        {
            Value = new Term[height, witdh];
            Width = witdh;
            Height = height;
        }

        public TermMatrix(Term[,] array)
        {
            Value = array;
            Width = array.GetLength(1);
            Height = array.GetLength(0);
        }

        private Term[,] Value { get; }

        public int Width { protected set; get; }
        public int Height { protected set; get; }

        public Term this[int row, int column]
        {
            set
            {
                row.Should().BeLessThan(Height);
                column.Should().BeLessThan(Width);
                Value[row, column] = value;
            }
            get
            {
                row.Should().BeLessThan(Height);
                column.Should().BeLessThan(Width);
                return Value[row, column];
            }
        }


        public Term[] GetRow(int row)
        {
            row.Should().BeLessThan(Height);
            return Enumerable.Range(0, Width).Select(c => Value[row, c]).ToArray();
        }

        public Term[] GetColumn(int column)
        {
            column.Should().BeLessThan(Width);
            return Enumerable.Range(0, Height).Select(r => Value[r, column]).ToArray();
        }

        public void SetRow(int row, Term[] terms)
        {
        }

        public void SetCoilumn(int column, Term[] terms)
        {
        }

        public override string ToString()
        {
            return $"TermMatrix:{Height},{Width}\r{Value}";
        }
    }
}