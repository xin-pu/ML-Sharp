using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using FluentAssertions;

namespace ML.Core.Data.Loader
{
    /// <summary>
    ///     A text loader for get IEnum<T> from txt file.
    /// </summary>
    public class TextLoader
    {
        public static Dataset<DataView> LoadDataSet<T>(string path, char[] splitChar, bool hasHeader = true)
            where T : DataView
        {
            /// Step 0 Precheck
            File.Exists(path).Should().BeTrue($"File {path} should exist.");

            /// Step 1 Read Stream to DataTable or Array
            using var stream = new StreamReader(path);
            var allline = stream.ReadToEnd()
                .Split('\r', '\n')
                .Where(a => !string.IsNullOrEmpty(a))
                .ToList();
            if (hasHeader)
                allline.RemoveAt(0);
            var alldata = allline.Select(l => l.Split(splitChar).ToArray()).ToArray();

            /// Step 2 Get Field Dict which have LoadColumn
            var fieldDict = GetFieldDict(typeof(T));

            /// Step 2 According LoadColumnAttribute Change to Data
            var datas = alldata.Select(single => GetData(typeof(T), fieldDict, single)).ToArray();

            /// Step 3 Return Dataset 
            return new Dataset<DataView>(datas);
        }

        public static Dataset<DataView> LoadDataSet<T>(string path, char splitChar, bool hasHeader = true)
            where T : DataView
        {
            return LoadDataSet<T>(path, new[] {splitChar}, hasHeader);
        }

        public static Dataset<DataView> LoadDataSet(string path, Type type, char[] splitChar, bool hasHeader)
        {
            /// Step 0 Precheck
            File.Exists(path).Should().BeTrue($"File {path} should exist.");

            /// Step 1 Read Stream to DataTable or Array
            using var stream = new StreamReader(path);
            var allline = stream.ReadToEnd()
                .Split('\r', '\n')
                .Where(a => !string.IsNullOrEmpty(a))
                .ToList();
            if (hasHeader)
                allline.RemoveAt(0);
            var alldata = allline.Select(l => l.Split(splitChar).ToArray()).ToArray();

            /// Step 2 Get Field Dict which have LoadColumn
            var fieldDict = GetFieldDict(type);

            /// Step 2 According LoadColumnAttribute Change to Data
            var datas = alldata.Select(single => GetData(type, fieldDict, single)).ToArray();

            /// Step 3 Return Dataset 
            return new Dataset<DataView>(datas);
        }

        private static Dictionary<FieldInfo, Range> GetFieldDict(Type type)
        {
            var fieldInfo = type.GetFields()
                .Where(a => a.CustomAttributes
                    .Any(attributeData => attributeData.AttributeType == typeof(LoadColumnAttribute)))
                .ToList();
            var dict = fieldInfo.ToDictionary(
                f => f,
                f => f.GetCustomAttribute<LoadColumnAttribute>().Range);
            return dict;
        }


        private static DataView GetData(Type classType, Dictionary<FieldInfo, Range> dict, string[] array)
        {
            var obj = Activator.CreateInstance(classType);
            dict.ToList().ForEach(p =>
            {
                var fieldInfo = p.Key;
                var range = p.Value;
                var type = fieldInfo.FieldType;

                if (range.Min == range.Max)
                {
                    var field = Convert.ChangeType(array[range.Min], type);
                    fieldInfo.SetValue(obj, field);
                }
                else if (type.IsArray && range.Max >= range.Min)
                {
                    var len = range.Max - range.Min + 1;
                    var arr = Activator.CreateInstance(type, len);
                    Enumerable.Range(0, len).ToList().ForEach(i =>
                    {
                        var field = Convert.ChangeType(array[range.Min + i], type.GetElementType()!);
                        type.GetMethod("Set")?.Invoke(arr, new[] {i, field});
                    });
                    fieldInfo.SetValue(obj, arr);
                }
            });

            return (DataView) obj;
        }
    }
}