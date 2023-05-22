using AutoDiff;
using FluentAssertions;
using Numpy;

namespace ML.Utility
{
    public class VariableMatrix
    {
        public VariableMatrix(int witdh, int height)
        {
            Value = new Variable[height, witdh];
            Width = witdh;
            Height = height;
        }


        public VariableMatrix(Variable[,] array)
        {
            Value = array;
            Width = array.GetLength(1);
            Height = array.GetLength(0);
        }

        private Variable[,] Value { get; }

        public int Width { protected set; get; }
        public int Height { protected set; get; }

        public Variable this[int row, int column]
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


        public Variable[] GetRow(int row)
        {
            row.Should().BeLessThan(Height);
            return Enumerable.Range(0, Width).Select(c => Value[row, c]).ToArray();
        }

        public Variable[] GetColumn(int column)
        {
            column.Should().BeLessThan(Width);
            return Enumerable.Range(0, Height).Select(r => Value[r, column]).ToArray();
        }


        public TermMatrix Power(double power)
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = TermBuilder.Power(this[r, c], power);
            return clone;
        }

        public TermMatrix Exp()
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = TermBuilder.Exp(this[r, c]);
            return clone;
        }

        public TermMatrix Log()
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = TermBuilder.Log(this[r, c]);
            return clone;
        }

        public TermMatrix Sigmoid()
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = TermOp.Sigmoid(this[r, c]);
            return clone;
        }

        public TermMatrix Tanh()
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = TermOp.Tanh(this[r, c]);
            return clone;
        }


        public TermMatrix Negation()
        {
            var clone = new TermMatrix(Width, Height);
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                clone[r, c] = -this[r, c];
            return clone;
        }

        public Term Sum()
        {
            var terms = new List<Term>();
            foreach (var r in Enumerable.Range(0, Height))
            foreach (var c in Enumerable.Range(0, Width))
                terms.Add(Value[r, c]);
            return TermBuilder.Sum(terms);
        }


        public Term Average()
        {
            return Sum() / (Width * Height);
        }


        public override string ToString()
        {
            return $"TermMatrix:{Height},{Width}\r{Value}";
        }

        public VariableMatrix Clone()
        {
            return new VariableMatrix((Variable[,]) Value.Clone());
        }


        #region operator

        public static TermMatrix operator +(VariableMatrix left, VariableMatrix right)
        {
            left.Width.Should().Be(right.Width);
            left.Height.Should().Be(right.Height);
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] += right[r, c];
            return clone;
        }

        public static TermMatrix operator +(VariableMatrix left, NDarray right)
        {
            left.Width.Should().Be(right.shape[1]);
            left.Height.Should().Be(right.shape[0]);
            var clone = new TermMatrix(left.Width, left.Height);
            var array = right.GetData<double>();
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] += array[r * left.Width + c];
            return clone;
        }


        public static TermMatrix operator +(VariableMatrix left, double right)
        {
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] += right;
            return clone;
        }

        public static TermMatrix operator -(VariableMatrix left, VariableMatrix right)
        {
            left.Width.Should().Be(right.Width);
            left.Height.Should().Be(right.Height);
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] -= right[r, c];
            return clone;
        }

        public static TermMatrix operator -(VariableMatrix left, NDarray right)
        {
            left.Width.Should().Be(right.shape[1]);
            left.Height.Should().Be(right.shape[0]);
            var clone = new TermMatrix(left.Width, left.Height);
            var array = right.GetData<double>();
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] -= array[r * left.Width + c];
            return clone;
        }

        public static TermMatrix operator -(VariableMatrix left, double right)
        {
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] -= right;
            return clone;
        }


        public static TermMatrix operator *(VariableMatrix left, VariableMatrix right)
        {
            left.Width.Should().Be(right.Width);
            left.Height.Should().Be(right.Height);
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] *= right[r, c];
            return clone;
        }

        public static TermMatrix operator *(VariableMatrix left, NDarray right)
        {
            left.Width.Should().Be(right.shape[1]);
            left.Height.Should().Be(right.shape[0]);
            var clone = new TermMatrix(left.Width, left.Height);
            var array = right.GetData<double>();
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] *= array[r * left.Width + c];
            return clone;
        }

        public static TermMatrix operator *(VariableMatrix left, double right)
        {
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] *= right;
            return clone;
        }

        public static TermMatrix operator /(VariableMatrix left, VariableMatrix right)
        {
            left.Width.Should().Be(right.Width);
            left.Height.Should().Be(right.Height);
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] /= right[r, c];
            return clone;
        }

        public static TermMatrix operator /(VariableMatrix left, NDarray right)
        {
            left.Width.Should().Be(right.shape[1]);
            left.Height.Should().Be(right.shape[0]);
            var clone = new TermMatrix(left.Width, left.Height);
            var array = right.GetData<double>();
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] /= array[r * left.Width + c];
            return clone;
        }

        public static TermMatrix operator /(VariableMatrix left, double right)
        {
            var clone = new TermMatrix(left.Width, left.Height);
            foreach (var r in Enumerable.Range(0, left.Height))
            foreach (var c in Enumerable.Range(0, left.Width))
                clone[r, c] /= right;
            return clone;
        }

        #endregion
    }
}