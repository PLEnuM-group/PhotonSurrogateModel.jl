using PhotonSurrogateModel
using Test

@testset "PhotonSurrogateModel.jl" begin
    @testset "NormalizingFlow" begin

        num_bins = 10
        x = -10:0.1:10
        params = randn(3 * num_bins + 1,  length(x))
        x_pos, y_pos, knot_slopes = constrain_spline_params(params, -5, 5)
        y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)
        xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)

        @test all(logdet .≈ -logdet_inv)
        @test all(isapprox.(x, xrt; atol=1E-5))

        x = CuArray(x)
        params = CuMatrix(params)
        x_pos, y_pos, knot_slopes = constrain_spline_params(params, -5, 5)
        y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)
        xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)

        @test all(Vector(logdet) .≈ -Vector(logdet_inv))
        @test all(isapprox.(Vector(x), Vector(xrt); atol=1E-5))


        p = randn(3 * num_bins + 1 + 2)
        p[end] = abs(p[end])

        function integrand(x)
            x = vec([x])                
            lpdf = eval_transformed_normal_logpdf(x, p, -5, 5)
            return exp(lpdf[1])
        end

        @test isapprox(quadgk(integrand, -10, 10)[1], 1, atol=1E-5)

    end
end
