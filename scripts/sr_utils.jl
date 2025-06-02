using SpecialFunctions
using LossFunctions
struct PoissonLoss <: LossFunctions.SupervisedLoss end

function (loss::PoissonLoss)(prediction, target)
    if prediction <= 0
        return Inf
    end

    log_likelihood = -prediction + target * log(prediction) - loggamma(target + 1)
    return -log_likelihood
end

struct LogLPLoss{P} <: LossFunctions.SupervisedLoss end

LogLPLoss(p::Number) = LogLPLoss{p}()

function (loss::LogLPLoss{P})(prediction, target) where {P}
    if prediction <= 0
        return Inf
    end

    return (abs(log(prediction+1) - log(target+1)))^P
end

const LogL1Loss = LogLPLoss{1}
const LogL2Loss = LogLPLoss{2}


struct Chi2Loss <: LossFunctions.SupervisedLoss end

function (loss::Chi2Loss)(prediction, target)
    return ((prediction - target)^2 / abs(prediction+eps()))
end

struct LogChi2Loss <: LossFunctions.SupervisedLoss end

function (loss::LogChi2Loss)(prediction, target)
    return ((log(prediction+1) - log(target+1))^2 / log(prediction+1))
end


function select_loss_func(loss_name)
    if loss_name == "poisson"
        loss_func = PoissonLoss()
    elseif loss_name == "logl2"
        loss_func = LogL2Loss()
    elseif loss_name == "logl1"
        loss_func = LogL1Loss()
    elseif loss_name == "l1"
        loss_func = L1DistLoss()
    elseif loss_name == "l2"
        loss_func = L2DistLoss()
    elseif loss_name == "chi2"
        loss_func = Chi2Loss()
    elseif loss_name == "logchi2"
        loss_func = LogChi2Loss()
    else
        error("Unknown loss function $(loss_name)")
    end
    return loss_func
end


exp_minus(x) = exp(-x)
one_over_square(x) = x^(-2)
expsq(x) = exp(x^2)
expabs(x) = exp(abs(x))
expnegsq(x) = exp(-x^2)

function literaltoreal(x)
    if SymbolicUtils.issym(x)
        return SymbolicUtils.Sym{Real}(nameof(x))
    elseif istree(x) && SymbolicUtils.symtype(x) <: LiteralReal
        return similarterm(x, operation(x), arguments(x), Real)
    else
        return x
    end
end