using System;

namespace ML.Core.Data.Loader
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class LoadColumnAttribute : Attribute
    {
        public int ColumnIndex;

        /// <summary>Maps member to specific field in text file.</summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public LoadColumnAttribute(int fieldIndex)
        {
            ColumnIndex = fieldIndex;
        }
    }
}