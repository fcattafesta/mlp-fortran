FC = gfortran
FCFLAGS = -J./mod -I./mod
SRCDIR = ./src
MODDIR = ./mod
OBJDIR = ./obj
BINDIR = ./bin

# VPATH specifies the directories to be searched for modules
VPATH = $(SRCDIR)/modules

# The wildcard function is used to generate a list of .f90 files in the modules directory
MODULES = $(wildcard $(SRCDIR)/modules/*.f90)

# The patsubst function is used to generate a list of .o files corresponding to the .f90 files
OBJS = $(patsubst $(SRCDIR)/modules/%.f90, $(OBJDIR)/%.o, $(MODULES))

all: directories train

train: $(OBJS) $(SRCDIR)/main.f90
	$(FC) $(FCFLAGS) -o $(BINDIR)/$@ $^

$(OBJDIR)/%.o: $(SRCDIR)/modules/%.f90
	$(FC) $(FCFLAGS) -c $< -o $@

directories: $(BINDIR) $(MODDIR) $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

$(MODDIR):
	mkdir -p $(MODDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -f $(BINDIR)/* $(MODDIR)/*.mod $(OBJDIR)/*.o