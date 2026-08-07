#include "xielu.h"
#include <torch/extension.h>

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class XIELUAutograd : public torch::autograd::Function<XIELUAutograd> {
public:
    static Tensor forward(
        AutogradContext* ctx,
        Tensor x,
        Tensor alpha_p,
        Tensor alpha_n,
        double beta,
        double eps
    ) {
        ctx->save_for_backward({x, alpha_p, alpha_n});
        ctx->saved_data["beta"] = beta;
        ctx->saved_data["eps"] = eps;

        return xielu_forward(x, alpha_p, alpha_n, beta, eps);
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        Tensor x = saved[0];
        Tensor alpha_p = saved[1];
        Tensor alpha_n = saved[2];
        double beta = ctx->saved_data["beta"].toDouble();
        double eps = ctx->saved_data["eps"].toDouble();

        Tensor grad_out = grad_outputs[0];
        if (!grad_out.is_contiguous()) {
            grad_out = grad_out.contiguous();
        }
        if (!x.is_contiguous()) {
            x = x.contiguous();
        }

        auto [gi, galpha_p, galpha_n] = xielu_backward(x, grad_out, alpha_p, alpha_n, beta, eps);
        Tensor undef;
        return {gi, galpha_p, galpha_n, undef, undef};
    }
};

Tensor xielu(const Tensor& x, const Tensor& alpha_p, const Tensor& alpha_n, double beta, double eps) {
    return XIELUAutograd::apply(x, alpha_p, alpha_n, beta, eps);
}

Tensor xielu_forward_meta(
    const Tensor& x,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    double beta,
    double eps
) {
    return torch::empty_like(x);
}

std::tuple<Tensor, Tensor, Tensor> xielu_backward_meta(
    const Tensor& x,
    const Tensor& go,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    double beta,
    double eps
) {
    return std::make_tuple(
        torch::empty_like(x),
        torch::empty({1}, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device())),
        torch::empty({1}, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()))
    );
}

Tensor xielu_meta(
    const Tensor& x,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    double beta,
    double eps
) {
    return torch::empty_like(x);
}

// Inference entry point for vLLM, which constructs torch.classes.xielu.XIELU()
// and calls .forward(x, alpha_p, alpha_n, beta, eps, with_vector_loads). The
// class is stateless: it forwards to the same kernel training uses, so both
// paths compute identical activations. `with_vector_loads` is a kernel hint
// upstream's implementation accepts; this kernel always uses its own load path.
struct XIELUInference : torch::CustomClassHolder {
    Tensor forward(
        const Tensor& x,
        const Tensor& alpha_p,
        const Tensor& alpha_n,
        double beta,
        double eps,
        bool with_vector_loads
    ) {
        (void)with_vector_loads;
        at::NoGradGuard no_grad;
        // The kernel indexes the buffer linearly, so views handed over by the
        // caller (vLLM reshapes activations before the call) must be packed.
        return xielu_forward(x.contiguous(), alpha_p.contiguous(), alpha_n.contiguous(), beta, eps);
    }
};

TORCH_LIBRARY(xielu, m) {
    m.def("forward(Tensor x, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> Tensor");
    m.def("backward(Tensor x, Tensor go, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> (Tensor, Tensor, Tensor)");
    m.def("xielu(Tensor x, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> Tensor");

    m.class_<XIELUInference>("XIELU")
        .def(torch::init<>())
        .def("forward", &XIELUInference::forward);
}

TORCH_LIBRARY_IMPL(xielu, CUDA, m) {
    m.impl("forward", &xielu_forward);
    m.impl("backward", &xielu_backward);
    m.impl("xielu", &xielu);
}

TORCH_LIBRARY_IMPL(xielu, Meta, m) {
    m.impl("forward", &xielu_forward_meta);
    m.impl("backward", &xielu_backward_meta);
    m.impl("xielu", &xielu_meta);
}

PYBIND11_MODULE(_xielu, m) {}
