using AutoDiff;

namespace ML.Utilty
{
    /// <summary>
    ///     Todo
    /// </summary>
    public class TermMatrix
    {
        public TermMatrix(int witdh, int height)
        {
        }

        public TermMatrix(Term[,] array)
        {
            Value = Value;
        }

        private Term[,] Value { get; }

        public int Width { protected set; get; }
        public int Height { protected set; get; }
    }
}