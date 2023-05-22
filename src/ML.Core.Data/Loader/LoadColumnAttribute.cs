namespace ML.Core.Data.Loader
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class LoadColumnAttribute : Attribute
    {
        public Range Range;

        /// <summary>Maps member to specific field in text file.</summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public LoadColumnAttribute(int fieldIndex)
        {
            Range = new Range(fieldIndex);
        }

        public LoadColumnAttribute(int min, int max)
        {
            Range = new Range(min, max);
        }
    }
}