include Makefile.conf


CFLAGS   += -pthread -O3 -g -std=c++14 
ifdef DEBUG
	CFLAGS += -fsanitize=address  -D_GLIBCXX_DEBUG 
else 
	CFLAGS += -DEIGEN_NO_DEBUG
endif

INCLUDE  += -I../../
LIBS     += 
SRCDIR   = ../../tasktorrent/src
OBJDIR   = ../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

default: all
all: rand_range

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

rand_range: rand_range.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)


clean:
	rm -f denseQR
	rm -f $(OBJDIR)/*.o
