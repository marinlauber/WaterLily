using Logging, LoggingExtras

function Logging.handle_message(logger::SimpleLogger,
                                lvl, msg, _mod, group, id, file, line;
                                kwargs...)
    # Write the formatted log message to logger.io
    println(logger.io, "[", lvl, "] ", msg)
end


ENV["JULIA_DEBUG"] = all

# fname = "testlog"
# to log iterations
# io = open(fname*".log", "w+")
# logger = SimpleLogger(io, Logging.Debug)
# global_logger(logger)

logger = FormatLogger("testlog.log"; append=false) do io, args
    println(io, "[", args.level, "] ", args.message)
end;
i=1
with_logger(logger) do
    @info "this is an info i=$i"
    @debug "         this is a debug i=$i"
end
