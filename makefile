# -*- mode: makefile-gmake -*-

override PETSCOPT_DIR := $(CURDIR)
include ./lib/petscopt/conf/variables
-include makefile.in

all-parallel: ruler := $(subst -,==========,--------)
all-parallel:
	-@echo $(ruler)
	-@echo "Building PETSCOPT (GNU Make - $(MAKE_NP) build jobs)"
	-@echo "Using PETSCOPT_DIR=$(PETSCOPT_DIR)"
	-@echo "Using PETSCOPT_ARCH=$(PETSCOPT_ARCH)"
	-@echo "Using PETSC_DIR=$(PETSC_DIR)"
	-@echo "Using PETSC_ARCH=$(PETSC_ARCH)"
	-@echo "Using fast=$(fast) debug=$(debug)"
	-@echo $(ruler)
	 @$(OMAKE) -j $(MAKE_NP) all
	-@echo $(ruler)
.PHONY: all-parallel

OBJDIR := $(PETSCOPT_ARCH)/obj
MODDIR := $(PETSCOPT_ARCH)/include
LIBDIR := $(abspath $(PETSCOPT_ARCH)/lib)

libpetscopt_shared := $(LIBDIR)/libpetscopt.$(SL_LINKER_SUFFIX)
libpetscopt_static := $(LIBDIR)/libpetscopt.$(AR_LIB_SUFFIX)
libpetscopt := $(if $(filter-out no,$(BUILDSHAREDLIB)),$(libpetscopt_shared),$(libpetscopt_static))
all : $(generated) $(libpetscopt)

ifeq ($(V),)           # Print help and short compile line
  quiet_HELP := "Use \"$(MAKE) V=1\" to see the verbose compile lines.\n"
  quiet = @printf $(quiet_HELP)$(eval quiet_HELP:=)"  %-10s %s\n" "$1$2" "$@"; $($1)
else ifeq ($(V),0)     # Same as previous, but do not print any help
  quiet = @printf "  %-10s %s\n" "$1$2" "$@"; $($1)
else                   # Show the full command line
  quiet = $($1)
endif

pcc = $(if $(findstring CONLY,$(PETSC_LANGUAGE)),CC,CXX)
COMPILE.cc = $(call quiet,$(pcc)) $(PCC_FLAGS) $(CFLAGS) $(CCPPFLAGS) $(C_DEPFLAGS) -c
COMPILE.cxx = $(call quiet,CXX) $(CXX_FLAGS) $(CFLAGS) $(CCPPFLAGS) $(CXX_DEPFLAGS) -c
ifneq ($(FC_MODULE_OUTPUT_FLAG),)
COMPILE.fc = $(call quiet,FC) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) $(FC_MODULE_OUTPUT_FLAG)$(MODDIR) -c
else
FCMOD = cd $(MODDIR) && $(FC)
COMPILE.fc = $(call quiet,FCMOD) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) -c
endif

generated := $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/lib/petscopt/conf/files

.SECONDEXPANSION: # to expand $$(@D)/.DIR

write-variable = @printf "override $2 = $($2)\n" >> $1;
petscopt-conf-root = $(PETSCOPT_ARCH)
config-petsc  := $(petscopt-conf-root)/lib/petscopt/conf/config-petsc
config-petscopt   := $(petscopt-conf-root)/lib/petscopt/conf/config-petscopt
$(config-petsc) : | $$(@D)/.DIR
	$(call write-variable,$@,PETSC_DIR)
	$(call write-variable,$@,PETSC_ARCH)
$(config-petscopt) : | $$(@D)/.DIR
	$(call write-variable,$@,fast)
	$(call write-variable,$@,debug)
config-clean :
	@$(RM) $(config-petsc) $(config-petscopt)
config-check :
	@if [ -e $(config-petsc) ]; then echo "Already configured" && false; fi
	@if [ -e $(config-petscopt)  ]; then echo "Already configured" && false; fi

config-petsc : $(config-petsc)
config-petscopt : $(config-petscopt)
config : config-petsc config-petscopt
.PHONY:  config-petsc config-petscopt config

$(generated) : | $$(@D)/.DIR
	$(PYTHON) $(PETSC_DIR)/config/gmakegen.py --petsc-arch=$(PETSC_ARCH) --pkg-dir=$(PETSCOPT_DIR)

-include $(generated)

langs := c
pkgs := ts
concatlang = $(foreach lang, $(langs), $(srcs-$(1).$(lang):src/%.$(lang)=$(OBJDIR)/%.o))
srcs.o := $(foreach pkg, $(pkgs), $(call concatlang,$(pkg)))

objects.o := $(srcs.o)
.SECONDARY: $(objects.o)

$(libpetscopt_static) : objs := $(objects.o)
$(libpetscopt_shared) : objs := $(objects.o)
$(libpetscopt_shared) : libs := $(PETSC_LIB)
$(libpetscopt_shared) : LDSL := CLINKER

%.$(SL_LINKER_SUFFIX) : $$(objs) | $$(@D)/.DIR
	$(call quiet,$(LDSL)) -shared -o $@ $^ $(libs)
ifneq ($(DSYMUTIL),true)
	$(call quiet,DSYMUTIL) $@
endif

%.$(AR_LIB_SUFFIX) : $$(objs) | $$(@D)/.DIR
ifeq ($(findstring win32fe lib,$(AR)),)
	@$(RM) $@
	$(call quiet,AR) $(AR_FLAGS) $@ $^
	$(call quiet,RANLIB) $@
else
	@$(RM) $@ $@.args
	@cygpath -w $^ > $@.args
	$(call quiet,AR) $(AR_FLAGS) $@ @$@.args
	@$(RM) $@.args
endif

$(OBJDIR)/%.o : src/%.c | $$(@D)/.DIR
	$(COMPILE.cc) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.cxx | $$(@D)/.DIR
	$(COMPILE.cxx) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.F90 | $$(@D)/.DIR $(MODDIR)/.DIR
	$(COMPILE.fc) $(abspath $<) -o $(if $(FCMOD),$(abspath $@),$@)

%/.DIR :
	@$(MKDIR) $(@D)
	@touch $@

.PRECIOUS: %/.DIR
.SUFFIXES: # Clear .SUFFIXES because we don't use implicit rules
.DELETE_ON_ERROR: # Delete likely-corrupt target file if rule fails
.PHONY: all check clean distclean install

#check :
#	@$(OMAKE) -C test/ check

clean :
	$(RM) -r $(OBJDIR) $(LIBDIR)/libpetscopt*.*
distclean :
	@echo "*** Deleting all build files ***"
	-$(RM) -r $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/

#PREFIX = /tmp/petscopt
#find-install-dir = \
#find $2 -type d -exec \
#install -m $1 -d "$(PREFIX)/{}" \;
#find-install = \
#find $2 -type f -name $3 -exec \
#install -m $1 "{}" "$(PREFIX)/$(if $4,$4,{})" \;
#install :
#	@echo "*** Installing PETSCOPT in PREFIX=$(PREFIX) ***"
#	@$(call find-install-dir,755,bin)
#	@$(call find-install-dir,755,include)
#	@$(call find-install-dir,755,lib)
#	@$(call find-install,755,bin,'petscopt-*')
#	@$(call find-install,644,include,'*.h')
#	@$(call find-install,644,lib/petscopt/app,'*')
#	@$(call find-install,644,lib/petscopt/conf,'*')
#	@$(call find-install,644,lib/petscopt/python,'*.py')
#	@$(call find-install,644,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(AR_LIB_SUFFIX)',lib)
#	@$(call find-install,755,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(SL_LINKER_SUFFIX)',lib)
#	@printf "override PETSCOPT_ARCH =\n" > $(PREFIX)/lib/petscopt/conf/arch;
#	@$(OMAKE) petscopt-conf-root=$(PREFIX) config;
#	@$(RM) $(PREFIX)/lib/petscopt/conf/.DIR;

# TAGS generation
alletags :
	-@$(PYTHON) $(PETSC_DIR)/lib/petsc/bin/maint/generateetags.py
deleteetags :
	-@$(RM) CTAGS TAGS
.PHONY: alletags deleteetags

# make print VAR=the-variable
print : ; @echo "$($(VAR))"
.PHONY: print
# make print-VARIABLE
print-% : ; @echo "$* = $($*)"
.PHONY: print-%


objects.d := $(objects.o:%.o=%.d)
# Tell make that objects.d are all up to date. Without
# this, the include below has quadratic complexity.
$(objects.d) : ;
-include $(objects.d)

# Handle test framework (also defines compilation lines for both lib and test)
include ./makefile.test        # This must be below the all target
