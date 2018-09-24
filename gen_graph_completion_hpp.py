import argparse


def make_completion_code(op_name,
                         attribute_list=[],
                         shape_inference_code='''
assert(node.input_name_list.size() == 1);
assert(node.output_name_list.size() == 1);
add_variable_to_table(output(0), dtype_of(input(0)), dims_of(input(0)));
''',
                         preprocess="",
                         postprocess=""):
    # attribute completion and definition
    attribute_completion_code_list = []
    attribute_definition_list = []
    for attribute in attribute_list:
        attr_name, attr_type, default_value = attribute
        inner_code = ''
        if default_value is None:
            inner_code = 'throw "error";'
        else:
            inner_code = '''
node.attribute_table.emplace(
    "{attr_name}", {default_value});
'''.format(attr_name=attr_name, default_value=default_value)

        attribute_completion_code = '''
{{
    auto found = node.attribute_table.find("{attr_name}");
    if(found == node.attribute_table.end()) {{
        {code}
    }}
}}
'''.format(attr_name=attr_name, attr_type=attr_type, code=inner_code)
        attribute_completion_code_list.append(attribute_completion_code)

        attribute_definition = '''
auto {attr_name} = get<{attr_type}>(node.attribute_table.at("{attr_name}"));
'''.format(attr_name=attr_name, attr_type=attr_type)
        attribute_definition_list.append(attribute_definition)
    # end for

    template = '''
if(node.op_type == "{op_name}") {{
    {preprocess}
    {attribute_completion_code}
    {{
        {attribute_definition}
        {shape_inference_code}
    }}
    {postprocess}
}}
'''
    return template.format(
        op_name=op_name,
        preprocess=preprocess,
        attribute_definition="\n".join(attribute_definition_list),
        shape_inference_code=shape_inference_code,
        postprocess=postprocess,
        attribute_completion_code="\n".join(
            attribute_completion_code_list))


def main():
    template = """
#ifndef MENOH_GRAPH_COMPLETION_HPP
#define MENOH_GRAPH_COMPLETION_HPP

#include <menoh/array.hpp>
#include <menoh/model_data.hpp>

namespace menoh_impl {{
    inline auto complete_model_data(model_data& model_data,
            std::unordered_map<std::string, array_profile> const&
                input_profile_table) {{
        using ints = std::vector<int>;
        std::unordered_map<std::string, array_profile> variable_profile_table(
            model_data.parameter_name_and_array_list.begin(),
            model_data.parameter_name_and_array_list.end());
        auto dims_of = [&model_data](std::string const& name){{
            auto found = variable_profile_table.find(name);
            if(found == variable_profile_table.end()) {{
                throw "error"; //TODO
            }}
            return found->second.dims();
        }};
        auto dtype_of = [&model_data](std::string const& name){{
            auto found = variable_profile_table.find(name);
            if(found == model_data.parameter_name_and_array_list.end()) {{
                throw "error"; //TODO
            }}
            return found->second.dtype();
        }};
        auto ndims_of = [&dims_of](std::string const& parameter_name) {{
            return dims_of(parameter_name).size();
        }};
        auto add_variable_to_table = [&variable_profile_table](
            std::string const& name,
            dtype_t dtype, ints const& dims){{
                variable_profile_table.emplace(
                    name, array_profile(dtype, dims));
            }};

        for(auto& node : model_data.node_list) {{
            auto input = [&node](int i){{
                return node.input_name_list.at(i);
            }};
            auto output = [&node](int i){{
                return node.output_name_list.at(i);
            }};
            {code}
        }}
    }}
}} // namespace menoh_impl

#endif // MENOH_GRAPH_COMPLETION_HPP
"""
    code_list = []
    code_list.append(make_completion_code("Abs"))
    code_list.append(make_completion_code("Add"))
    code_list.append(
        make_completion_code("AveragePool", [
            ("count_include_pad", "int", "0"),
            ("kernel_shape", "ints", None),
            ("pads", "ints", "ints(ndims_of(input(1))-2, 0)"),
            ("strides", "ints", None),
        ], '''
add_variable_to_table(dtype_of(input(0)), calc_2d_output_dims(
    dims_of(input(0)), kernel_shape, strides, pads));
'''))
    code_list.append(
        make_completion_code("BatchNorm", [
            ("epsilon", "float", "1.e-05f"),
            ("momentum", "float", "0.9f"),
            ("spatial", "int", "1"),
        ]))
    code_list.append(
        make_completion_code("Concat", [
            ("axis", "int", None),
        ], '''
#TODO
'''))
    code_list.append(
        make_completion_code(
            "Conv", [
                ("dilations", "ints", "ints(kernel_ndims, 1)"),
                ("group", "int", "1"),
                ("kernel_shape", "ints", "kernel_shape"),
                ("pads", "ints", "ints(kernel_ndims*2, 0)"),
                ("strides", "ints", "ints(kernel_ndims, 1)"),
            ], '''
add_variable_to_table(dtype_of(input(0)), calc_2d_output_dims(
    dims_of(input(0)), kernel_shape, strides, pads));
''',
            preprocess='''
auto kernel_ndims = ndims_of(input(1))-2;
auto weights_shape = dims_of(input(1));
auto kernel_shape = ints(weights_shape.begin()+2, weights_shape.end());
'''))
    code_list.append(
        make_completion_code(
            "ConvTranspose",
            [
                ("dilations", "ints", None),
                ("group", "int", "1"),
                ("kernel_shape", "ints", "kernel_shape"),
                ("output_padding", "ints", None),
                #("output_shape", "ints", None),
                #("pads", "ints", None),
                ("strides", "ints", "ints(kernel_ndims, 1)"),
            ], '''
#TODO
''',
            preprocess='''
auto kernel_ndims = ndims_of(input(1))-2;
auto weights_shape = dims_of(input(1));
auto kernel_shape = ints(weights_shape.begin()+2, weights_shape.end());
''',
            postprocess='''
{
    auto found = node.attribute_table.find("output_shape");
    if(found == node.attribute_table.end() &&
       node.attribute_table.find("pads") == node.attribute_table.end()) {
        throw "error";
    }
    if(found != node.attribute_table.end()) {
        auto output_shape = get<ints>(found->second);
        /* [dim0_begin, dim1_begin, ... , dim0_end, dim1_end, ..., ...] */
        ints pads(kernel_ndims*2, 0);
        auto output_padding =
            get<ints>(node.attribute_table.at("output_padding"));
        auto strides = get<ints>(node.attribute_table.at("strides"));
        auto input_profile = input_profile_table.at(input(0));
        ints input_size(input_profile.dims().begin()+2,
                        input_profile.dims().end());

        for(int i = 0; i < kernel_ndims; ++i) {
            auto total_padding = strides[i] * (input_size[i] - 1)
                + output_padding[i] + kernel_shape[i] - output_shape[i];
            pads[i] = total_padding - (total_padding/2);
            pads[i+kernel_ndims] = (total_padding/2);
        }

        node.attribute_table["pads"] = pads;
    }
}
'''))
    code_list.append(make_completion_code("Elu", [("alpha", "float", "1.f")]))
    code_list.append(
        make_completion_code("Gemm", [
            ("alpha", "float", "1.f"),
            ("beta", "float", "1.f"),
            ("transA", "int", "0"),
            ("transB", "int", "0"),
        ], '''
#TODO
'''))
    code_list.append(
        make_completion_code("LeakyRelu", [("alpha", "float", "0.01f")]))
    code_list.append(
        make_completion_code("LRN", [
            ("alpha", "float", "0.0001f"),
            ("beta", "float", "0.75f"),
            ("bias", "float", "1.0f"),
            ("size", "float", None),
        ]))
    code_list.append(
        make_completion_code("MaxPool", [
            ("kernel_shape", "ints", None),
            ("pads", "ints", "ints(ndims_of(input(1))-2, 0)"),
            ("storage_order", "int", "0"),
            ("strides", "ints", None),
        ], '''
add_variable_to_table(dtype_of(input(0)), calc_2d_output_dims(
    dims_of(input(0)), kernel_shape, strides, pads));
'''))
    code_list.append(make_completion_code("Relu"))
    code_list.append(make_completion_code("Softmax", [("axis", "int", "1")]))
    code_list.append(make_completion_code("Sum"))
    code_list.append(make_completion_code("Sqrt"))
    code_list.append(make_completion_code("Tanh"))
    code_list.append(
        make_completion_code("Transpose", [
            ("perm", "ints", '''
[&](){{
    ints perm(ndims_of(input(0)));
    for(int i = 0; i < perm.size(); ++i) {{
        perm.at(i) = perm.size()-i-1;
    }}
    return perm;
}}()
'''.format()),
        ], '''
#TODO
'''))
    print(template.format(code="\n".join(code_list)))


if __name__ == "__main__":
    main()
